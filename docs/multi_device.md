# Multi-hardware Adaptation

Paddle ecosystem relies on the contributions of developers and users. We warmly welcome contributions to adapt more models for multi-hardware support in Paddle.

## 1. Supported Hardware List

| Task Type | Model Name |   NVIDIA   | KUNLUNXIN | HYGON | Tecorigin |   MetaX   |
|-----------|------------|------------|-----------|-------|-----------|-----------|
| MLIP(Machine Learning Interatomic Potential) | [CHGNet](../interatomic_potentials/configs/chgnet/README.md)            | ✅ |  |  |  | |
| MLIP(Machine Learning Interatomic Potential) | [MatterSim](../interatomic_potentials/configs/mattersim/README.md)      | ✅ |  |  |  | |
| PP(Property Prediction)                      | [MEGNet](../property_prediction/configs/megnet/README.md)               | ✅ |  |  |  | |
| PP(Property Prediction)                      | [DimeNet++](../property_prediction/configs/dimenet++/README.md)         | ✅ |  |  |  | |
| PP(Property Prediction)                      | [ComFormer](../property_prediction/configs/comformer/README.md)         | ✅ |  |  |  | |
| SG(Structure Generation)                     | [DiffCSP](../structure_generation/configs/diffcsp/README.md)            | ✅ |  |  |  | |
| SG(Structure Generation)                     | [MatterGen](../structure_generation/configs/mattergen/README.md)        | ✅ |  |  |  | |
| SE(Spectrum Elucidation)                     | [DiffNMR](../spectrum_elucidation/configs/diffnmr/README.md)            | ✅ |  |  |  | |


## 2. How to Contribute

We provide reference accuracy based on NVIDIA CUDA training and corresponding pre-trained model weights at the beginning of our public case documentation. If you need to run the models on specific hardware, please follow these steps:

1.If your hardware type has not yet been integrated into PaddlePaddle, you can refer to the official documentation of PaddleCustomDevice to integrate it into the Paddle framework. If your hardware type has been integrated into PaddlePaddle but has not yet been added to PaddleMaterials' hardware support list, please add your hardware type in the tast clarrification README document..

2.Prepare the necessary dataset according to the steps provided in the case documentation.

3.If the model documentation provides model training commands, perform full training on your hardware, save the training logs, record the best model accuracy, and the best model weights. These are usually automatically saved in the case folder during training.

4.If the model documentation provides model evaluation commands, evaluate the best model saved in step 3 on your hardware, save the evaluation logs, and record the evaluation accuracy. These are usually automatically saved in the case folder during evaluation.

5.If the model documentation provides model export and inference commands, follow these commands to verify whether model export and inference can be executed normally on the new hardware and whether the inference results align with CUDA's results.

6.After completing the above steps, you can add your hardware support information (✅) to the corresponding model in the table.  And submit a PR to PaddleMaterials. Your PR should include at least the following:
a.A usage guide document for running the model in your hardware environment
b.The best model weights file saved during training (.pdparams file).
c.Training/evaluation logs (.log files).
d.Software versions used for validating model accuracy, including but not limited to:
    d.1 PaddlePaddle version
    d.2 PaddleCustomDevice version (if applicable)
e.Machine environment details used for validating model accuracy, including but not limited to:
    e.1 Chip model
    e.2 System version
    e.3 Hardware driver version
    e.4 Operator library version, etc.

## 3. More Referenced Documents
* [PaddleUserGuide(ch)](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/index_cn.html)
* [PaddleSupportedHardware(ch)](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/hardware_support/index_cn.html)
* [PaddleCustomDevice](https://github.com/PaddlePaddle/PaddleCustomDevice)