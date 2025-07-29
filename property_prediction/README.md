# IOMPP-Inorganic Materials Property Prediction

## 1.Introduction

Inorganic Materials Property Prediction (IMPP) targets rapid, first-principles-level estimation of key crystalline properties—formation energy, band gap, elastic moduli, ionic conductivity, and more—without performing new density-functional-theory calculations. The workflow mirrors modern ML interatomic-potential pipelines but shifts the label space from forces to scalar and tensor observables. Starting from crystal structure files (CIF), an automated converter builds atom–bond graphs enriched with chemical descriptors and symmetry-aware positional encodings. Equivariant graph neural networks, or transformer-based variants, are then trained on tens of thousands of reference entries. By collapsing months of high-throughput DFT time into minutes of GPU inference, IMPP empowers data-driven discovery of semiconductors, catalysts and functional

## 2.Models Matrix

| **Supported Functions**                      | **[MEGNet](./configs/megnet/README.md)** | **[Comfomer](./configs/comformer/README.md)** | **GemNet** | **[DimeNet++](./configs/dimenet++/README.md)** | **InfGCN** |
| -------------------------------------------- | :--------------------------------------: | :-------------------------------------------: | :--------: | :--------------------------------------------: | :--------: |
| **Forward Prediction · Materials Properties**|                                          |                                               |            |                                                |            |
| Formation energy                             |                    ✅                    |                       ✅                      |     🚧     |                       ✅                       |     —      |
| Band gap                                     |                    ✅                    |                       ✅                      |     🚧     |                       ✅                       |     —      |
| Bulk modulus                                 |                    ✅                    |                       ✅                      |     🚧     |                       ✅                       |     —      |
| Shear modulus                                |                    ✅                    |                       ✅                      |     🚧     |                       ✅                       |     —      |
| Young’s modulus                              |                    ✅                    |                       ✅                      |     🚧     |                       ✅                       |     —      |
| Adsorption energy                            |                    🚧                    |                       🚧                      |     🚧     |                       🚧                       |     —      |
| Electron density                             |                     —                    |                       —                       |     —      |                       —                        |     🚧     |
| **ML Capabilities · Training**               |                                          |                                               |            |                                                |            |
| Single-GPU                                   |                    ✅                    |                       ✅                      |     🚧     |                       ✅                       |     🚧     |
| Distributed training                         |                    ✅                    |                       ✅                      |     🚧     |                       ✅                       |     —      |
| Mixed precision (AMP)                        |                    —                     |                       —                       |     —      |                       —                        |     —      |
| Fine-tuning                                  |                    ✅                    |                       ✅                      |     🚧     |                       ✅                       |     —      |
| Uncertainty / Active Learning                |                    —                     |                       —                       |     —      |                       —                        |     —      |
| Dynamic→Static graphs                        |                    —                     |                       —                       |     —      |                       —                        |     —      |
| Compiler (CINN) opt.                         |                    —                     |                       —                       |     —      |                       —                        |     —      |
| **ML Capabilities · Predict**                |                                          |                                               |            |                                                |            |
| Distillation / Pruning                       |                    —                     |                       —                       |     —      |                       —                        |     —      |
| Standard inference                           |                    ✅                    |                       ✅                      |     🚧     |                       ✅                       |     🚧     |
| Distributed inference                        |                    —                     |                       —                       |     —      |                       —                        |     —      |
| Compiler-level inference                     |                    —                     |                       —                       |     —      |                       —                        |     —      |
| **Datasets**                                 |                                          |                                               |            |                                                |            |
| **Materials Project**                        |                                          |                                               |            |                                                |            |
| MP2024                                       |                    ✅                    |                       ✅                      |     —      |                       —                        |     —      |
| MP2020                                       |                    ✅                    |                       ✅                      |     —      |                       —                        |     —      |
| MP2018                                       |                    ✅                    |                       ✅                      |     🚧     |                       —                        |     —      |
| **JARVIS**                                   |                                          |                                               |            |                                                |            |
| dft_2d                                       |                    ✅                    |                       ✅                      |     —      |                       ✅                       |     —      |
| dft_3d                                       |                    ✅                    |                       ✅                      |     —      |                       —                        |     —      |
| **Alexandria**                               |                                          |                                               |            |                                                |            |
| pbe_2d                                       |                    ✅                    |                       ✅                      |     🚧     |                       —                        |     —      |
| **ML2DDB🌟**                                 |                    ✅                    |                       ✅                      |     ✅     |                       ✅                       |     —      |

**Notice**:🌟 represent originate research work published from paddlematerials toolkit
