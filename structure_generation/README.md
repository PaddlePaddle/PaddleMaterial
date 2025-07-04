# IOMSG-Inorganic Material Structure Generation

## 1.Introduction

The inorganic material structure generation (IMSG) task tackles the inverse-design challenge of creating entirely new crystal structures that satisfy stability and functional constraints without exhaustive enumeration. Models first embed known crystals into symmetry-aware latent spaces—fractional-coordinate graphs, Wyckoff-sequence tokens, or E(3)-equivariant voxel fields. Generators—diffusion models, graph-autoregressive Transformers, or symmetry-equivariant GANs—sample this space. Running on a single GPU, the framework can propose over a thousand candidate crystal structures per minute, dramatically lowering the trial-and-error cost of discovering scintillators, solid-state electrolytes, and high-entropy compounds. Combined with a rapid, tiered screening funnel—machine-learning potential relaxation, energy threshold filtering, and final DFT refinement—this keeps computation affordable and tightly couples theory with experiment.

## 2.Models Matrix

| **Supported Functions** | **DiffCSP** | **MatterGen** |
|-------------|-----------|--------------|
| **Structure Generation** |||
| &emsp;Random Sample | ✅ | ✅ |
| &emsp;Condition Sample  | ✅ | ✅ |
| **ML Capabilities · Training** |||
| &emsp;Single-GPU        | ✅ | ✅ |
| &emsp;Distributed Train | ✅ | ✅ |
| &emsp;Mixed Precision   | - | - |
| &emsp;Fine-tuning       | ✅ | ✅ |
| &emsp;Uncertainty / Active-Learning | - | - |
| &emsp;Dynamic→Static    | - | - |
| &emsp;Compiler CINN | - | - |
| **ML Capabilities · Predict** |||
| &emsp;Distillation / Pruning | - | - | 
| &emsp;Standard inference     | ✅ | ✅ |
| &emsp;Distributed inference  | - | - |
| &emsp;Compiler CINN | - | - |
| **Dataset** |||
|**Material Project**|||
| &emsp;MP20 | ✅ | ✅ |
|**Hrbrid**|||
| &emsp;ALEX MP20 | - | ✅ |




