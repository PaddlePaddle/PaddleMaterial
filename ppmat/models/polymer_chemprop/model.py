import paddle
import paddle.nn as nn

from ppmat.models.polymer_chemprop.featurization import Featurization_parameters
from ppmat.models.polymer_chemprop.mpn import MPN
from ppmat.models.polymer_chemprop.nn_utils import get_activation_function
from ppmat.models.polymer_chemprop.nn_utils import initialize_weights


class PolymerChempropModel(nn.Layer):
    """A PolymerChempropModel is a message passing network followed by feed-forward layers
    for molecular property prediction, ported from polymer-chemprop (PyTorch) to PaddlePaddle.
    """

    def __init__(
        self,
        hidden_size: int = 300,
        depth: int = 3,
        dropout: float = 0.0,
        activation: str = "ReLU",
        undirected: bool = False,
        atom_messages: bool = False,
        bias: bool = False,
        aggregation: str = "mean",
        aggregation_norm: int = 100,
        ffn_num_layers: int = 2,
        ffn_hidden_size: int = 300,
        num_tasks: int = 1,
        dataset_type: str = "regression",
        number_of_molecules: int = 1,
        mpn_shared: bool = False,
        features_only: bool = False,
        features_size: int = 0,
        use_input_features: bool = False,
        atom_descriptors: str = None,
        atom_descriptors_size: int = 0,
        multiclass_num_classes: int = 3,
        property_name: str = "target",
        featurization_config: Featurization_parameters = None,
    ):
        super(PolymerChempropModel, self).__init__()

        self.dataset_type = dataset_type
        self.classification = dataset_type == "classification"
        self.multiclass = dataset_type == "multiclass"
        self.num_tasks = num_tasks
        self.multiclass_num_classes = multiclass_num_classes
        self.property_name = property_name

        self.output_size = num_tasks
        if self.multiclass:
            self.output_size *= multiclass_num_classes

        # Build encoder
        self.encoder = MPN(
            hidden_size=hidden_size,
            bias=bias,
            depth=depth,
            dropout=dropout,
            undirected=undirected,
            atom_messages=atom_messages,
            aggregation=aggregation,
            aggregation_norm=aggregation_norm,
            activation=activation,
            features_only=features_only,
            use_input_features=use_input_features,
            atom_descriptors=atom_descriptors,
            atom_descriptors_size=atom_descriptors_size,
            number_of_molecules=number_of_molecules,
            mpn_shared=mpn_shared,
            featurization_config=featurization_config,
        )

        # Build FFN
        if features_only:
            first_linear_dim = features_size
        else:
            first_linear_dim = hidden_size * number_of_molecules
            if use_input_features:
                first_linear_dim += features_size

        if atom_descriptors == "descriptor":
            first_linear_dim += atom_descriptors_size

        dropout_layer = nn.Dropout(dropout)
        act = get_activation_function(activation)

        if ffn_num_layers == 1:
            ffn = [dropout_layer, nn.Linear(first_linear_dim, self.output_size)]
        else:
            ffn = [dropout_layer, nn.Linear(first_linear_dim, ffn_hidden_size)]
            for _ in range(ffn_num_layers - 2):
                ffn.extend(
                    [
                        act,
                        dropout_layer,
                        nn.Linear(ffn_hidden_size, ffn_hidden_size),
                    ]
                )
            ffn.extend(
                [
                    act,
                    dropout_layer,
                    nn.Linear(ffn_hidden_size, self.output_size),
                ]
            )

        self.ffn = nn.Sequential(*ffn)

        initialize_weights(self)

    def forward(self, data, return_loss=True, return_prediction=True):
        """
        Forward pass following PaddleMaterials convention.

        :param data: A dict with keys: batch_graphs, labels, label_mask, features, atom_descriptors_batch.
        :param return_loss: Whether to compute and return the loss.
        :param return_prediction: Whether to return predictions.
        :return: A dict with 'loss_dict' and 'pred_dict'.
        """
        assert (
            return_loss or return_prediction
        ), "At least one of return_loss or return_prediction must be True."

        batch_graphs = data["batch_graphs"]
        features = data.get("features")
        atom_descriptors_batch = data.get("atom_descriptors_batch")

        output = self.ffn(
            self.encoder(
                batch_graphs,
                features_batch=features,
                atom_descriptors_batch=atom_descriptors_batch,
            )
        )

        # Multiclass reshape
        if self.multiclass:
            output = output.reshape([output.shape[0], -1, self.multiclass_num_classes])

        out = {"loss_dict": {}, "pred_dict": {}}

        if return_loss and "labels" in data:
            labels = data["labels"]
            label_mask = data.get("label_mask")

            if self.dataset_type == "regression":
                if label_mask is not None:
                    loss = (
                        paddle.nn.functional.mse_loss(
                            output * label_mask, labels * label_mask, reduction="sum"
                        )
                        / label_mask.sum()
                    )
                else:
                    loss = paddle.nn.functional.mse_loss(output, labels)
            elif self.dataset_type == "classification":
                if label_mask is not None:
                    loss = (
                        paddle.nn.functional.binary_cross_entropy_with_logits(
                            output * label_mask, labels * label_mask, reduction="sum"
                        )
                        / label_mask.sum()
                    )
                else:
                    loss = paddle.nn.functional.binary_cross_entropy_with_logits(
                        output, labels
                    )
            elif self.dataset_type == "multiclass":
                labels_long = labels.astype("int64")
                loss = paddle.nn.functional.cross_entropy(
                    output.reshape([-1, self.multiclass_num_classes]),
                    labels_long.reshape([-1]),
                    reduction="mean",
                )
            else:
                raise ValueError(f'Dataset type "{self.dataset_type}" not supported.')

            out["loss_dict"]["loss"] = loss

        if return_prediction:
            pred = output
            if self.dataset_type == "classification":
                pred = paddle.nn.functional.sigmoid(output)
            elif self.multiclass:
                pred = paddle.nn.functional.softmax(output, axis=2)
            out["pred_dict"][self.property_name] = pred

        return out

    @paddle.no_grad()
    def predict(self, batch_graphs, features=None, atom_descriptors_batch=None):
        """
        Convenience method for inference.

        :param batch_graphs: A list of BatchMolGraph.
        :param features: Optional features batch.
        :param atom_descriptors_batch: Optional atom descriptors batch.
        :return: A dict mapping property_name to predictions (numpy array).
        """
        self.eval()
        data = {
            "batch_graphs": batch_graphs,
            "features": features,
            "atom_descriptors_batch": atom_descriptors_batch,
        }
        result = self.forward(data, return_loss=False, return_prediction=True)
        pred = result["pred_dict"][self.property_name].numpy()
        return {self.property_name: pred}
