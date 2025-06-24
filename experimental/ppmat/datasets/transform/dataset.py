import paddle
from paddle.io import DataLoader
from typing import Union, List

__all__ = [
    "no_scaling",
    "mean_std_scaling",
    "rmsd_scaling",
    "custom_scaling",
    ]


def no_scaling(
        train_loader: DataLoader,
        target: Union[str, List[str]],
        **kwargs
    ):
    return 0.0, 1.0


def mean_std_scaling(
        train_loader: DataLoader,
        target: Union[str, List[str]],
        **kwargs
    ):
    target_list = []

    if isinstance(target, list):
        if len(target) == 1:
            target = target[0]
        else:
            raise NotImplementedError("Current mean_std_scaling only supports single-target data")

    for _, batch_data in enumerate(train_loader):
        target_list.append(batch_data[target])
    graph_target = paddle.concat(target_list, axis=0)        # [total_n_graphs]
    mean = paddle.mean(graph_target).numpy()
    std = paddle.std(graph_target).numpy()
    return mean, std


def rmsd_scaling(
        train_loader: DataLoader,
        target: Union[str, List[str]],
        **kwargs
    ):
    raise NotImplementedError("rmsd_scaling has been removed")
    # else:
    #     for batch in train_dataset:
    #         target_list.append(batch.target)                    # {[n_graphs*n_atoms,3], 
    #     vector_target = torch.cat(target_list, dim=0)           # {[total_n_graphs*n_atoms,3], }
    #     mean = to_numpy(torch.mean(vector_target)).item()
    #     std = to_numpy(torch.std(vector_target)).item()


def custom_scaling(
        train_loader: DataLoader,
        target: Union[str, List[str]],
        **kwargs
    ):
    if len(kwargs) == 0:
        raise ValueError("custom_scaling requires at least one parameter (e.g., 'mean' and 'std', or 'rmsd').")
    
    if "mean" in kwargs and "std" in kwargs:
        return float(kwargs["mean"]), float(kwargs["std"])
    elif "rmsd" in kwargs:
        return 0.0, float(kwargs["rmsd"])
    else:
        raise ValueError("Required keyword arguments not found. Please check or modify the 'custom_scaling' function.")
