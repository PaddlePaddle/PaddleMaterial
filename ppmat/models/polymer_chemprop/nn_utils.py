import paddle
import paddle.nn as nn


def index_select_ND(source: paddle.Tensor, index: paddle.Tensor) -> paddle.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.

    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
                  indices to select from source.
    :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
             features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.shape
    suffix_dim = source.shape[1:]
    final_size = list(index_size) + list(suffix_dim)
    target = paddle.index_select(source, index.reshape([-1]), axis=0)
    target = target.reshape(final_size)
    return target


def get_activation_function(activation: str) -> nn.Layer:
    """
    Gets an activation function module given the name of the activation.

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')


def initialize_weights(model: nn.Layer) -> None:
    """
    Initializes the weights of a model in place.

    :param model: A PaddlePaddle model.
    """
    for param in model.parameters():
        if len(param.shape) == 1:
            nn.initializer.Constant(value=0)(param)
        else:
            nn.initializer.XavierNormal()(param)
