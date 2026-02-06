# 本目录是针对 chemeleon2 的 单元测试说明

## 目录说明
- test_loss 是测试paddle版本 和 原始版本的精度差异的目录【可以在当前项目运行】
- raw_infer_data 是原始的 hspark1212/chemeleon2 commit-id:63769eb4b962278cb0576da1ac7e0cc10d36a3e8 版本项目中，增加的测试脚本， 可以产出工程里的输入和输出数据，方便paddle版本对比【只能在chemeleon2原始项目中运行】。



## test_loss 目录介绍

### test_model_loss_with_raw.py 脚本
可以通过pytest进行测试，需要提起准备好权重文件，权重文件在：https://aistudio.baidu.com/modelsdetail/43734?modelId=43734

## raw_infer_data 目录介绍

### create_input_output_npz.py 脚本

create_input_output_npz.py 文件需要在 hspark1212/chemeleon2 commit-id:63769eb4b962278cb0576da1ac7e0cc10d36a3e8 项目中运行；

运行后，会产生一系列的npz文件，其中 input 字样的是输入数据；output 字样是输出数据；

我们可以拿同样的 input的输入、同样的随机数 让paddle 框架来生成晶体结构获取结果，和原版的output输出的值之间进行对比，观测精度差异。

npz的格式化输出脚本如下：

```python

import numpy as np


def to_python_list(arr, max_dim=4):
    """将 numpy 数组转换为 Python 列表格式"""
    if arr.ndim == 0:
        return arr.item()
    if arr.ndim == 1:
        return arr.tolist()
    # 递归处理多维数组
    return to_python_list(arr, max_dim - 1) if max_dim > 1 else arr.tolist()


def print_array(arr, max_dim=4, indent=0):
    """以 Python 数组风格打印数组"""
    if isinstance(arr, np.ndarray):
        if arr.ndim == 0:
            print(str(arr.item()), end='')
        elif arr.ndim == 1:
            print('[', end='')
            for i, val in enumerate(arr):
                if i < min(4, len(arr)):
                    print_array(val, max_dim - 1, 0)
                    if i < min(4, len(arr)) - 1:
                        print(', ', end='')
            if len(arr) > 4:
                print(', ...]', end='')
            else:
                print(']', end='')
        else:
            print('[', end='')
            for i in range(min(4, len(arr))):
                print_array(arr[i], max_dim - 1, 0)
                if i < min(4, len(arr)) - 1:
                    print(', ', end='')
            if len(arr) > 4:
                print(', ...]', end='')
            else:
                print(']', end='')
    else:
        # 标量 - 保留小数点后8位
        if isinstance(arr, (np.integer, np.floating)):
            val = float(arr)
            # 整数不显示小数点
            if val == int(val):
                print(int(val), end='')
            else:
                # 保留小数点后8位
                print(f"{val:.8f}", end='')
        else:
            print(arr, end='')


def print_npz(file_path, max_dim=4):
    data = np.load(file_path)

    print(f"文件: {file_path}")
    print("=" * 50)

    for key in data.keys():
        arr = data[key]
        print(f"\n[{key}]")
        print(f"  shape: {arr.shape}, dtype: {arr.dtype}")
        print(f"  值: ", end='')
        print_array(arr, max_dim)
        print()


if __name__ == "__main__":
    # 手动指定 npz 文件路径
    npz_file = "outputs/precision_test/vae_decoder_input_z_full.npz"

    # 可选: 设置最大输出维度，默认 4
    max_output_dim = 4

    print_npz(npz_file, max_output_dim)

```