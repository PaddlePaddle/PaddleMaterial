import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from ppmat.model.mace import MACE

class MACETrainer:
    """Trainer for MACE model"""
    def __init__(self, config):
        self.config = config
        self.model = MACE(
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_basis=config['num_basis'],
            r_max=config['r_max'],
            num_elements=config['num_elements']
        )
        self.optimizer = optim.Adam(
            parameters=self.model.parameters(),
            learning_rate=config['learning_rate']
        )
        self.loss_fn = nn.MSELoss()
    def train_step(self, batch):
        """Single training step"""
        atomic_numbers, positions, energies, forces = batch
        
        # Forward pass
        pred_energy, pred_forces = self.model(atomic_numbers, positions)
        
        # Compute loss
        energy_loss = self.loss_fn(pred_energy, energies)
        force_loss = self.loss_fn(pred_forces, forces)
        total_loss = energy_loss + self.config['force_weight'] * force_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.clear_grad()
        
        return total_loss.item()
    def evaluate(self, data_loader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        with paddle.no_grad():
            for batch in data_loader:
                atomic_numbers, positions, energies, forces = batch
                pred_energy, pred_forces = self.model(atomic_numbers, positions)
                energy_loss = self.loss_fn(pred_energy, energies)
                force_loss = self.loss_fn(pred_forces, forces)
                batch_loss = energy_loss + self.config['force_weight'] * force_loss
                total_loss += batch_loss.item()
        self.model.train()
        return total_loss / len(data_loader)
    def save_model(self, path):
        """Save model"""
        paddle.save(self.model.state_dict(), path)
    def load_model(self, path):
        """Load model"""
        state_dict = paddle.load(path)
        self.model.set_state_dict(state_dict)
