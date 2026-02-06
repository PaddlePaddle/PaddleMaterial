import paddle


def to_dense_batch(x, batch_idx, max_num_nodes=None):
    """
    将 batch 格式的数据转换为 dense batch 格式，并生成 padding mask
    
    Args:
        x: 特征张量 [N, D]，N 是总原子数，D 是特征维度
        batch_idx: batch 索引 [N]，指示每个原子属于哪个结构
        max_num_nodes: 最大节点数，如果为 None 则自动计算
        
    Returns:
        x_dense: dense 格式的特征张量 [B, max_num_nodes, D]
        mask: padding mask [B, max_num_nodes]，True 表示有效位置，False 表示 padding
    """
    batch_size = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 1
    
    num_nodes = paddle.zeros([batch_size], dtype='int64')
    for i in range(batch_size):
        num_nodes[i] = (batch_idx == i).sum()
    
    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max().item())
    
    feat_dim = x.shape[-1]
    x_dense = paddle.zeros([batch_size, max_num_nodes, feat_dim], dtype=x.dtype)
    mask = paddle.zeros([batch_size, max_num_nodes], dtype='bool')
    
    cumsum = paddle.concat([paddle.zeros([1], dtype='int64'), 
                           paddle.cumsum(num_nodes, axis=0)[:-1]])
    
    for i in range(batch_size):
        start = int(cumsum[i].item())
        end = start + int(num_nodes[i].item())
        n = int(num_nodes[i].item())
        x_dense[i, :n] = x[start:end]
        mask[i, :n] = True
    
    return x_dense, mask

