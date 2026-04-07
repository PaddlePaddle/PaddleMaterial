import paddle
from ppmat.model.mace import MACE

class MACEPredictor:
    """Predictor for MACE model"""
    def __init__(self, model_path, config):
        self.config = config
        self.model = MACE(
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_basis=config['num_basis'],
            r_max=config['r_max'],
            num_elements=config['num_elements']
        )
        # Load model weights
        state_dict = paddle.load(model_path)
        self.model.set_state_dict(state_dict)
        self.model.eval()
    def predict(self, atomic_numbers, positions, cell=None, pbc=False):
        """Predict energy and forces"""
        with paddle.no_grad():
            energy, forces = self.model(atomic_numbers, positions, cell, pbc)
        return energy.numpy(), forces.numpy()
    def predict_batch(self, batch):
        """Predict batch of structures"""
        energies = []
        forces_list = []
        for atomic_numbers, positions, cell, pbc in batch:
            energy, forces = self.predict(atomic_numbers, positions, cell, pbc)
            energies.append(energy)
            forces_list.append(forces)
        return energies, forces_list
