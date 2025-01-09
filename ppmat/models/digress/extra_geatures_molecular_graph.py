import paddle

from ppmat.utils.digressutils import PlaceHolder


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
        返回 (X, E, y) 格式的 PlaceHolder
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

        return PlaceHolder(X=x_cat, E=extra_edge_attr, y=weight)


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
        """
        max_weight: 用于将分子总质量做归一化
        atom_weights: dict, 每种原子的原子量，如 {'H':1, 'C':12, 'O':16, ...}
        """
        self.max_weight = max_weight
        # 将 atom_weights 的 values 做成 paddle.Tensor
        self.atom_weight_list = paddle.to_tensor(
            list(atom_weights.values()), dtype="float32"
        )

    def __call__(self, noisy_data):
        """
        计算分子的总原子量 (bs, 1)，并除以 self.max_weight 进行归一化。
        """
        X_t = noisy_data["X_t"]  # (bs, n, dx) => one-hot
        # 取原子类型索引 => (bs, n)
        atom_type_idx = paddle.argmax(X_t, axis=-1)

        # 根据 atom_type_idx 取出对应原子量 => (bs, n)
        # 需先把 self.atom_weight_list broadcast 到和 atom_type_idx 兼容
        # 更简单方法：paddle.gather(self.atom_weight_list, atom_type_idx, axis=0) 也可行
        # 但 paddle.gather 需要自己写 batch 版，这里可使用直接索引:
        # 注意: 直接索引可能需先把 atom_type_idx 转 numpy，再转回来, 这里做个简单处理:

        # (bs, n) flatten => (bs*n,)
        shape_bsn = atom_type_idx.shape
        flattened_idx = paddle.reshape(atom_type_idx, [-1])  # (bs*n,)
        # gather => (bs*n,)
        gathered_weights = paddle.gather(self.atom_weight_list, flattened_idx, axis=0)
        # reshape => (bs, n)
        X_weights = paddle.reshape(gathered_weights, shape_bsn)

        # 分子总质量 => (bs,)
        mw = paddle.sum(X_weights, axis=-1)
        # => (bs,1)
        mw = paddle.unsqueeze(mw, axis=-1)

        # 归一化 + 与 X_t 保持相同 dtype
        return mw.astype(X_t.dtype) / self.max_weight
