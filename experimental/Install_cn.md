
# Installation

[English](./install.md)

## 1. 安装说明

我们推荐使用conda虚拟环境来管理依赖包，你可以通过安装[Miniforge](https://github.com/conda-forge/miniforge)使用conda。

### 1.1 创建虚拟环境
创建一个新的conda虚拟环境，并激活环境：

    conda create -n ppmat python=3.10
    conda activate ppmat

目前我们在python 3.10环境下进行开发，因此建议使用python 3.10或者更高的版本。

### 1.2 安装PaddlePaddle
根据你的cuda版本安装对应版本的PaddlePaddle，具体安装命令可参考[PaddlePaddle官网](https://www.paddlepaddle.org.cn/install/quick)。我们推荐安装PaddlePaddle >= 3.1或者develop版本。

例如，对于cuda12.6环境，安装paddlepaddle-gpu版本：

     python -m pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

安装完毕之后，运行以下命令，验证 Paddle 是否安装成功。

    python -c "import paddle; paddle.utils.run_check()"

如果出现 PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now. 信息，说明已成功安装。

### 1.3 源码安装PaddleMaterial：

    # clone PaddleMaterial
    git clone https://github.com/PaddlePaddle/PaddleMaterial.git

    # 切换到PaddleMaterial目录
    cd PaddleMaterial

    # 以可编辑模式安装PaddleMaterial
    pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple


## 2. 运行示例

使用 MegNet 模型预测材料属性：

    python property_prediction/predict.py --model_name='megnet_mp2018_train_60k_e_form' --weights_name='best.pdparams' --cif_file_path='./property_prediction/example_data/cifs/'

使用 MatterSim 模型预测能量和力：

    python interatomic_potentials/predict.py --model_name='mattersim_1M' --weights_name='mattersim-v1.0.0-1M_model.pdparams' --cif_file_path='./interatomic_potentials/example_data/cifs/'

更多的使用说明可以参考[Get Started](./get_started.md)。
