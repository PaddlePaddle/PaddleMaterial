# Installation 🔧

[简体中文](./Install_cn.md)

## 1. Installation Instructions

We recommend using a conda virtual environment to manage dependencies. You can install conda via [Miniforge](https://github.com/conda-forge/miniforge).

### 1.1 Create Virtual Environment
Create and activate a new conda virtual environment:

    conda create -n ppmat python=3.10
    conda activate ppmat

We currently develop under Python 3.10 environment and recommend using Python 3.10 or newer.

### 1.2 Install PaddlePaddle
Install the appropriate PaddlePaddle version based on your CUDA version. Refer to the [PaddlePaddle Official Website](https://www.paddlepaddle.org.cn/install/quick) for installation commands. We recommend installing PaddlePaddle version >= 3.1 or the develop version.

For example, in a CUDA 12.6 environment, install the paddlepaddle-gpu version:

    python -m pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

After installation, verify the installation with:

    python -c "import paddle; paddle.utils.run_check()"

If you see "PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.", the installation was successful.

### 1.3 Install PaddleMaterial from Source:

    # Clone PaddleMaterial repository
    git clone https://github.com/PaddlePaddle/PaddleMaterial.git

    # Navigate to PaddleMaterial directory
    cd PaddleMaterial

    # Install dependencies
    pip install --upgrade pip setuptools wheel
    pip install setuptools_scm
    pip install Cython

    # Install in editable mode
    pip install -e .

## 2. Run Examples

Predict material properties using the MegNet model:

    python property_prediction/predict.py --model_name='megnet_mp2018_train_60k_e_form' --weights_name='best.pdparams' --cif_file_path='./property_prediction/example_data/cifs/'

Predict energy and forces using the MatterSim model:

    python interatomic_potentials/predict.py --model_name='mattersim_1M' --weights_name='mattersim-v1.0.0-1M_model.pdparams' --cif_file_path='./interatomic_potentials/example_data/cifs/'

For more usage instructions, refer to the [Get Started](./get_started.md) documentation.
