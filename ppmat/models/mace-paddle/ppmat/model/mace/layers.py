import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .utils import radial_basis

class RadialBasisFunction(nn.Layer):
    """Radial basis function layer"""
    def __init__(self, r_max, num_basis):
        super().__init__()
        self.r_max = r_max
        self.num_basis = num_basis
    def forward(self, r):
        return radial_basis(r, self.r_max, self.num_basis)

class EquivariantLayer(nn.Layer):
    """Equivariant message passing layer"""
    def __init__(self, hidden_dim, num_basis, r_max):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_basis = num_basis
        self.r_max = r_max
        
        # Radial basis function
        self.rbf = RadialBasisFunction(r_max, num_basis)
        
        # Message passing weights
        self.W_message = nn.Linear(num_basis + hidden_dim, hidden_dim)
        
        # Update weights
        self.W_update = nn.Linear(hidden_dim * 2, hidden_dim)
    def forward(self, x, edge_indices, edge_distances):
        # Get radial basis features
        rbf_features = self.rbf(edge_distances)
        
        # Get source and target nodes
        src, dst = edge_indices
        
        # Message passing
        x_src = x[src]
        message = paddle.concat([x_src, rbf_features], axis=-1)
        message = self.W_message(message)
        message = F.relu(message)
        
        # Aggregate messages
        aggregated = paddle.zeros_like(x)
        aggregated = paddle.scatter(aggregated, dst.unsqueeze(-1), message, axis=0, reduce='sum')
        
        # Update node features
        updated = paddle.concat([x, aggregated], axis=-1)
        updated = self.W_update(updated)
        updated = F.relu(updated)
        
        return updated
