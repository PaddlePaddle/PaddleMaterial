import paddle

from .utils import digressutils as utils


class ExtraMolecularFeatures:
    def __init__(self, dataset_infos):
        """
        dataset_infos 中通常包含:
          - remove_h: 是否移除氢 (bool)
          - valencies: dict or list, 各原子对应的最大价/可能价
          - max_weight: 用于归一化分子量的最大值
          - atom_weights: dict, 记录每种原子的原子量 (如 {'H':1, 'C':12, ...})
        """
        self.charge = ChargeFeature(
            remove_h=dataset_infos.remove_h, valencies=dataset_infos.valencies
        )
        self.valency = ValencyFeature()
        self.weight = WeightFeature(
            max_weight=dataset_infos.max_weight, atom_weights=dataset_infos.atom_weights
        )

    def __call__(self, noisy_data):
        """
        计算并拼接分子/原子额外特征:
         - 原子电荷 (charge)
         - 原子实际价 (valency)
         - 分子重量 (weight)
        返回 (X, E, y) 格式的 utils.PlaceHolder
        """
        # charge / valency => (bs, n, 1)
        charge = paddle.unsqueeze(self.charge(noisy_data), axis=-1)
        valency = paddle.unsqueeze(self.valency(noisy_data), axis=-1)
        # weight => (bs, 1)
        weight = self.weight(noisy_data)

        # 边级额外特征默认为空 (bs, n, n, 0)
        E_t = noisy_data["E_t"]
        extra_edge_attr = paddle.zeros(shape=E_t.shape[:-1] + [0], dtype=E_t.dtype)

        # 将电荷与价拼接到原子特征 X 的最后一维: (bs, n, 2)
        x_cat = paddle.concat([charge, valency], axis=-1)

        return utils.PlaceHolder(X=x_cat, E=extra_edge_attr, y=weight)


class ChargeFeature:
    def __init__(self, remove_h, valencies):
        self.remove_h = remove_h
        self.valencies = valencies

    def __call__(self, noisy_data):
        """
        估算每个原子的净电荷 = (理想价态 - 当前键合数)。
        bond_orders = [0, 1, 2, 3, 1.5] 分别表示：无键 / 单键 / 双键 / 三键 / 芳香键。
        """
        E_t = noisy_data["E_t"]
        dtype_ = E_t.dtype

        bond_orders = paddle.to_tensor([0, 1, 2, 3, 1.5], dtype=dtype_)
        bond_orders = paddle.reshape(bond_orders, [1, 1, 1, -1])  # (1,1,1,5)

        # E_t * bond_orders => (bs, n, n, de)，取 argmax => (bs, n, n)，再 sum => (bs,n)
        weighted_E = E_t * bond_orders
        current_valencies = paddle.argmax(weighted_E, axis=-1)  # (bs, n, n)
        current_valencies = paddle.sum(current_valencies, axis=-1)  # (bs, n)

        # 计算理想价态
        X_t = noisy_data["X_t"]
        valency_tensor = paddle.to_tensor(
            self.valencies, dtype=X_t.dtype
        )  # shape (dx,)
        valency_tensor = paddle.reshape(valency_tensor, [1, 1, -1])  # (1,1,dx)
        X_val = X_t * valency_tensor  # (bs, n, dx)
        normal_valencies = paddle.argmax(X_val, axis=-1)  # (bs, n)

        # 电荷 = (理想价态 - 当前键合数)
        charge = normal_valencies - current_valencies
        return charge.astype(X_t.dtype)


class ValencyFeature:
    def __init__(self):
        pass

    def __call__(self, noisy_data):
        """
        计算每个原子的实际价态 (仅由当前键类型决定)。
        """
        E_t = noisy_data["E_t"]
        dtype_ = E_t.dtype
        orders = paddle.to_tensor([0, 1, 2, 3, 1.5], dtype=dtype_)
        orders = paddle.reshape(orders, [1, 1, 1, -1])  # (1,1,1,5)

        E_weighted = E_t * orders  # (bs, n, n, de)
        valencies = paddle.argmax(E_weighted, axis=-1)  # (bs, n, n)
        valencies = paddle.sum(valencies, axis=-1)  # (bs, n)

        X_t = noisy_data["X_t"]
        return valencies.astype(X_t.dtype)


class WeightFeature:
    def __init__(self, max_weight, atom_weights):
        """Set weights for each type of atom based on their atomic weight.

        Args:
            max_weight (Int): Max weight of atom
                to normalize the total molecular mass.
            atom_weights (Dict): Atomic weight of each atom.
        """
        self.max_weight = max_weight
        self.atom_weight_list = paddle.to_tensor(
            list(atom_weights.values()), dtype="float32"
        )

    def __call__(self, noisy_data):
        X = paddle.argmax(noisy_data["X_t"], axis=-1)  # (bs, n)
        X_weights = self.atom_weight_list[X]  # (bs, n)
        return (
            X_weights.sum(axis=-1).unsqueeze(-1).astype(noisy_data["X_t"].dtype)
            / self.max_weight
        )  # (bs, 1)
