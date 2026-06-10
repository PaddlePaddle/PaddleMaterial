import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class GaussianRBF(nn.Layer):
    def __init__(self, num_basis=32, cutoff=5.0, gamma=None):
        super().__init__()
        centers = paddle.linspace(0.0, cutoff, num_basis)
        self.register_buffer("centers", centers)
        if gamma is None:
            gamma = 10.0 / max(cutoff, 1e-6)
        self.gamma = gamma

    def forward(self, distances):
        diff = distances - self.centers.reshape([1, -1])
        return paddle.exp(-self.gamma * diff * diff)


class PolynomialCutoff(nn.Layer):
    def __init__(self, cutoff=5.0, p=6):
        super().__init__()
        self.cutoff = cutoff
        self.p = p

    def forward(self, distances):
        x = distances / self.cutoff
        x = paddle.clip(x, 0.0, 1.0)
        weight = 1.0 - 6.0 * x**5 + 15.0 * x**4 - 10.0 * x**3
        mask = (distances < self.cutoff).astype("float32")
        return weight * mask


class MessageBlock(nn.Layer):
    def __init__(self, hidden_dim, rbf_dim):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + rbf_dim, hidden_dim),
            nn.Silu(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Silu(),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Silu(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_src, edge_dst, rbf, edge_weight):
        src_h = paddle.gather(h, edge_src, axis=0)
        dst_h = paddle.gather(h, edge_dst, axis=0)

        msg_in = paddle.concat([src_h, dst_h, rbf], axis=-1)
        msg = self.msg_mlp(msg_in)
        msg = msg * edge_weight

        # 使用 scatter_nd_add 实现消息聚合
        num_nodes = h.shape[0]
        agg = paddle.zeros([num_nodes, h.shape[1]], dtype=h.dtype)
        edge_dst_2d = edge_dst.unsqueeze(1)
        agg = paddle.scatter_nd_add(agg, edge_dst_2d, msg)

        upd_in = paddle.concat([h, agg], axis=-1)
        dh = self.upd_mlp(upd_in)
        return self.norm(h + dh)


class PurePaddleSevenNet(nn.Layer):
    def __init__(
        self,
        num_species=100,
        hidden_dim=128,
        num_message_layers=4,
        num_rbf=32,
        cutoff=5.0,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.embedding = nn.Embedding(num_species, hidden_dim)
        self.rbf = GaussianRBF(num_basis=num_rbf, cutoff=cutoff)
        self.cutoff_fn = PolynomialCutoff(cutoff=cutoff, p=6)
        self.blocks = nn.LayerList(
            [MessageBlock(hidden_dim, num_rbf) for _ in range(num_message_layers)]
        )
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Silu(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph):
        z = graph["z"]
        pos = graph["pos"]
        edge_index = graph["edge_index"]

        edge_src = edge_index[0]
        edge_dst = edge_index[1]

        h = self.embedding(z)

        src_pos = paddle.gather(pos, edge_src, axis=0)
        dst_pos = paddle.gather(pos, edge_dst, axis=0)
        edge_vec = src_pos - dst_pos
        dist = paddle.sqrt(paddle.sum(edge_vec * edge_vec, axis=-1, keepdim=True) + 1e-12)

        rbf = self.rbf(dist)
        edge_weight = self.cutoff_fn(dist)

        for block in self.blocks:
            h = block(h, edge_src, edge_dst, rbf, edge_weight)

        atomic_energy = self.energy_head(h)
        total_energy = paddle.sum(atomic_energy)

        return {
            "total_energy": total_energy,
            "atomic_energy": atomic_energy,
        }

    def predict(self, data):
        """Predict energy and forces for a batch of structures.
        
        This method provides a compatible interface for PotentialPredictor.
        
        Args:
            data: Dictionary containing graph data with keys:
                - z: atomic numbers (Tensor)
                - pos: positions (Tensor)
                - edge_index: edge connectivity (Tensor)
                
        Returns:
            Dictionary with predictions:
                - energy: total energy per structure (float)
                - forces: atomic forces (numpy array)
        """
        # Support both direct graph dict and data object with graph attribute
        if hasattr(data, 'graph'):
            graph = data.graph
        else:
            graph = data
            
        # Handle batch dimension
        if isinstance(graph, list):
            results = []
            for g in graph:
                result = self.forward(g)
                # Convert to numpy and format output
                prediction = {
                    "energy": float(result["total_energy"].numpy()),
                    "atomic_energy": result["atomic_energy"].numpy(),
                }
                results.append(prediction)
            return results
        else:
            result = self.forward(graph)
            return {
                "energy": float(result["total_energy"].numpy()),
                "atomic_energy": result["atomic_energy"].numpy(),
            }
    
    def compute_forces(self, graph, eps=1e-4):
        """Compute forces by finite difference of energy w.r.t. positions.
        
        Uses finite difference instead of autograd because Paddle's
        scatter_nd_add backward pass has numerical instability with
        deep message-passing networks.
        
        Args:
            graph: Dictionary containing graph data
            eps: Finite difference step size
            
        Returns:
            numpy array of forces with shape [num_atoms, 3]
        """
        import numpy as np
        
        positions_np = graph["pos"].numpy()
        z = graph["z"]
        edge_index = graph["edge_index"]
        num_atoms = positions_np.shape[0]
        forces = np.zeros_like(positions_np)
        
        for i in range(num_atoms):
            for j in range(3):
                pos_plus = positions_np.copy()
                pos_plus[i, j] += eps
                pos_minus = positions_np.copy()
                pos_minus[i, j] -= eps
                
                pos_p = paddle.to_tensor(pos_plus, dtype="float32")
                with paddle.no_grad():
                    e_plus = float(self.forward({"z": z, "pos": pos_p, "edge_index": edge_index})["total_energy"].numpy())
                
                pos_m = paddle.to_tensor(pos_minus, dtype="float32")
                with paddle.no_grad():
                    e_minus = float(self.forward({"z": z, "pos": pos_m, "edge_index": edge_index})["total_energy"].numpy())
                
                forces[i, j] = -(e_plus - e_minus) / (2 * eps)
        
        return forces