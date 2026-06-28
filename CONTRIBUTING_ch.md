# Contributing to PaddleMaterials

感谢你对 PaddleMaterials 的关注与贡献！

本文面向线上仓库 [PaddlePaddle/PaddleMaterials](https://github.com/PaddlePaddle/PaddleMaterials) 的贡献。PaddleMaterials 当前是基于 PaddlePaddle 的 AI4Materials 工具库，核心 Python 包为 `ppmat`，模型权重通常使用 `.pdparams`。

## 1. 开发原则

提交到 PaddleMaterials 的代码不仅需要能运行，还需要满足以下项目要求：

- 符合仓库现有任务划分与目录结构。
- 符合 repo 已有代码逻辑、配置风格和调用方式。
- 按 README 可以完成主要训练、评测、推理或采样流程。
- 文档、配置、代码实现和复现结果保持一致。
- 优先保证可维护、可复现、可扩展。
- 不明显破坏已有模型、数据集和任务入口。

对于模型复现类 PR，本项目更关注：

- 是否接入到了正确的任务目录。
- 是否沿用了 PaddleMaterials 现有的数据流、配置流和入口脚本。
- 是否复用了 `ppmat` 中已有的 model、dataset、trainer、metric、scheduler、predictor、sampler 等公共能力。
- 是否补齐了 README、配置、测试或最小验证说明。
- 是否与现有模块风格一致，而不是简单搬运外部仓库代码。

## 2. 仓库结构与放置规范

请严格按照仓库现有结构放置代码，不要随意新建平行体系。

当前主要任务目录包括：

- `property_prediction/`：性质预测任务（PP）。
- `structure_generation/`：结构生成任务（SG）。
- `interatomic_potentials/`：机器学习原子间势函数任务（MLIP）。
- `electronic_structure/`：电子结构任务（ES）。
- `spectrum_elucidation/`：谱图解析任务（SE）。
- `spectrum_enhancement/`：谱图或显微图像增强任务（SPEN）。
- `ppmatSim/`：基于模型或势函数的材料模拟流程。
- `research/`：研究性流程或实验性工作。
- `jointContribution/`：联合贡献或外部协同代码区域。该目录当前被 pre-commit 排除，新主线能力不建议默认放在这里，除非维护者明确要求。

公共实现主要放在：

- `ppmat/models/`：模型、图构建器和模型构建入口。
- `ppmat/datasets/`：数据集、数据构建、collate、transform 和 dataloader 构建入口。
- `ppmat/trainer/`：训练流程和训练状态。
- `ppmat/losses/`、`ppmat/metrics/`、`ppmat/optimizer/`、`ppmat/schedulers/`：损失、指标、优化器和调度器。
- `ppmat/predictor/`、`ppmat/sampler/`：公共推理和采样封装。
- `ppmat/calculator/`：ASE 等外部接口。
- `ppmat/utils/`：通用工具。
- `test/`：项目测试或验证脚本。
- `docs/`：项目文档。

## 3. 新模型接入

新增模型时，一般应至少涉及以下位置中的一部分：

- `ppmat/models/<model_name>/`：模型主体、图构建器或模型内部工具。
- `ppmat/models/__init__.py`：导入模型类，补充 `__all__`，有预训练权重时再补充 `MODEL_REGISTRY`。
- 对应任务目录下的 `configs/<model_name>/`：模型配置与 README。
- 对应任务目录下的 `train.py`、`predict.py`、`sample.py` 或相关入口适配。
- 必要时补充 `test/` 下测试或最小验证脚本。

请不要把完整模型逻辑直接堆在任务脚本里，也不要绕开 `ppmat` 现有抽象单独再造一套训练框架。

新增模型时请注意：

- 模型主体应使用 PaddlePaddle API，通常继承 `paddle.nn.Layer`。
- 配置中继续使用 `__class_name__` 和 `__init_params__` 约定。
- 优先复用现有 `Trainer`、`Dataset`、`Optimizer`、`Metric`、`Predict`、`Sample` 等配置结构。
- 预训练权重命名与现有习惯保持一致，常见后缀为 `.pdparams`，例如 `best.pdparams`、`latest.pdparams`。
- 当前仓库已经存在 `ppmat/predictor/` 和 `ppmat/sampler/` 公共封装。新增推理器或采样器时，应优先复用或扩展公共实现；确需任务特化时，应对齐对应任务目录已有入口风格。
- 如果修改公共 predictor 或 sampler，需要验证相关已有模型不被破坏。

来自其他框架或其他仓库的代码不能简单复制后直接提交，必须完成 PaddleMaterials 风格适配，包括：

- PaddlePaddle API 适配。
- 参数组织方式统一。
- 配置命名方式统一。
- 训练、推理、采样接口统一。
- 文档风格统一。
- 依赖和许可证说明清楚。

## 4. 新数据集接入

新增数据集一般应放在 `ppmat/datasets/`。如果数据处理逻辑较复杂，可以在该目录下新建子目录组织实现。

数据集实现应尽量复用现有 dataset 风格，保持：

- 初始化参数风格一致。
- 数据划分逻辑清晰。
- 字段命名清楚。
- 返回值结构稳定。
- 与现有 dataloader、collate、transform 兼容。
- 与下游模型所需字段匹配。
- 不在 dataset 内写过多任务特判逻辑。
- 不把预处理、训练逻辑和 dataset 强耦合。

数据集接入通常需要：

- 在 `ppmat/datasets/` 下新增数据集类。
- 在 `ppmat/datasets/__init__.py` 中导入数据集类，使现有 `build_dataloader` 能通过配置中的 `__class_name__` 找到它。
- 必要时补充 dataset info、transform、collate 或 graph build 逻辑。
- 如已有工厂函数、工具类或数据处理逻辑可以复用，请优先使用现有封装。

数据集适配类 PR 应优先参考仓库已有数据集实现，而不是直接照搬外部仓库写法。

数据集 README 至少说明：

- 数据集名称。
- 来源链接和许可限制。
- 原始格式。
- 下载方式。
- 预处理方式。
- 划分方式。
- 标签含义、单位和字段名。
- 与配置中路径、缓存路径的对应关系。

数据集适配 PR 至少应满足：

- 数据集类可实例化。
- 数据能够被正常读取。
- 样本字段与下游模型匹配。
- 按 README 能跑通基本流程。
- 文档与代码一致。

## 5. 工具脚本与转换脚本

`test/`、`tools/` 或任务目录下的工具脚本，只应放置：

- 用户需要明确调用的正式工具。
- 权重转换、数据转换、评测、结果对齐等确有长期维护意义的脚本。
- 有 README 或命令说明的迁移工具。

不应提交：

- 临时调试脚本。
- 只在作者本地环境下可用的脚本。
- 含硬编码个人路径、私有机器名或私有凭据的实验脚本。
- 一次性对齐脚本但没有说明文档的文件。
- 大日志、缓存、训练输出和无说明的中间文件。

## 6. PaddleMaterials 特定要求

### 6.1 必须沿用仓库已有入口与组织方式

新增内容应尽量接入现有流程，而不是另起炉灶。包括但不限于：

- 训练入口。
- 推理入口。
- 采样入口。
- 配置组织方式。
- 数据集注册和调用方式。
- 模型调用接口。
- 权重下载和加载方式。

如果现有任务目录已经存在同类模型或同类任务，请优先对齐该目录下已有实现。

例如：

- 晶体或分子性质预测任务，优先参考 `property_prediction/` 下已有模型与配置组织。
- 机器学习原子间势函数任务，优先参考 `interatomic_potentials/` 下已有模型与 README 组织。
- 结构生成任务，优先参考 `structure_generation/` 下 MatterGen、DiffCSP 的配置和采样入口。
- 谱图解析任务，优先参考 `spectrum_elucidation/` 下 DiffNMR 的训练和采样流程。
- 谱图或显微图像增强任务，优先参考 `spectrum_enhancement/` 下 SFIN 的配置和预测入口。
- 数据集适配，优先参考 `ppmat/datasets/` 中已有实现。

### 6.2 不接受仓库外风格强行迁入

来自其他框架或其他仓库的代码，不能简单复制后直接提交。

必须完成 PaddleMaterials 风格适配，包括：

- PaddlePaddle API 适配，不在主线功能中强依赖其他深度学习框架。
- 参数组织方式统一。
- 配置命名方式统一。
- 训练、推理、采样接口统一。
- 文档风格统一。
- 许可证、引用来源和第三方代码边界说明清楚。

### 6.3 使用 Paddle 官方正式版本

- 必须使用 PaddlePaddle 官方正式发布版本。
- 不得依赖 Paddle develop 版本作为唯一可运行环境。
- 若对 Paddle 版本有最低要求，需在 README 中明确写明。
- 若依赖特定 CUDA、Python、paddle_scatter、pgl、pymatgen、ase 等版本，也应一并说明。

对于 PR 验收，若代码只能在 Paddle develop 版本或私有环境下运行，一般不视为满足合入条件。

## 7. README 与文档要求

PaddleMaterials 的贡献，尤其是模型复现类和数据集适配类 PR，必须保证 README 可用。

README 至少应包含：

- 任务简介。
- 模型或数据集简介。
- 数据准备方式。
- 环境依赖和已验证版本。
- 训练命令。
- 评测命令。
- 推理或采样命令，如适用。
- 关键配置说明。
- 参考结果。
- 预训练权重或数据下载链接，如适用。
- 参考论文、官方实现或数据来源链接。

README 必须满足：

- 按照 README 操作，可以完成主要流程。
- 文档与代码一致。
- 命令能直接从仓库根目录复制运行，避免隐式前置条件。
- 不省略关键步骤。
- 不允许 README 写一套、代码实现另一套。

对于模型复现类 PR，建议在 README 中写清：

- 复现目标。
- 所用数据集版本。
- 关键超参数。
- 训练轮数、batch size、学习率。
- 硬件环境。
- 评测指标。
- 与原论文或参考实现的结果对比。
- 偏差范围与原因说明，如有。

若暂时无法完整对齐原论文结果，也应明确说明当前完成到什么程度，而不是模糊表述“支持该模型”。

## 8. 配置文件要求

配置文件应与 PaddleMaterials 现有 YAML 风格保持一致。

当前常见顶层字段包括：

- `Global`
- `Trainer`
- `Model`
- `Optimizer`
- `Metric`
- `Dataset`
- `Predict`
- `Sample` 或 `Sampler`
- `Loss`，如任务需要

配置内容应清晰完整，至少建议包含：

- 模型构建参数。
- 数据集路径、字段、transform、graph converter 或预处理参数。
- dataloader、sampler、batch size、num workers。
- optimizer 和 lr scheduler。
- loss 和 metric。
- trainer、predictor 或 sampler 相关设置。
- save、log、eval 相关设置。
- pretrained model 或 checkpoint 设置，如适用。

配置默认应尽量可运行。提交的配置不要求都是大规模训练配置，但至少应保证：

- 能正常解析。
- 能构建模型。
- 能构建数据集，或明确说明数据下载与准备方式。
- 能启动最小训练、评测、推理或采样流程。

命名建议体现任务、模型名、数据集名和训练目标，例如：

- `megnet_mp2018_train_60k_e_form.yaml`
- `chgnet_oc20_s2ef_energy.yaml`
- `dimenet++_mp2018_train_60k_band_gap.yaml`
- `diffcsp_mp20.yaml`
- `mattergen_alex_mp20_dft_band_gap.yaml`
- `infgcn_qm9.yaml`
- `sfin_haadf_enhance.yaml`

避免使用难以理解的命名，例如：

- `final_new.yaml`
- `test2.yaml`
- `run_ok.yaml`
- `debug.yaml`

## 9. 数据集文件和预训练模型权重

不建议把大体积数据集、预训练权重、缓存文件或训练输出直接提交到 git。

数据集或模型适配 PR 应提供验收所需的数据集文件或预训练模型权重。可以先将文件上传到网盘，并在 PR 中提供链接，或按项目约定 @ reviewer 协助上传到 BCE。

维护者或 reviewer 将数据集文件和预训练模型权重上传到 BCE 后，贡献者需要将正式链接添加到对应位置，例如：

- README 的下载说明。
- 配置中的默认路径或说明。
- `ppmat/models/__init__.py` 中的 `MODEL_REGISTRY`，如属于预训练模型。

请同时说明：

- 文件名和大小。
- 解压后的目录结构。
- checksum，建议提供。
- 数据或权重来源。
- 许可限制。

## 10. 测试与验收要求

PR 合入前，至少应满足核心功能的最小验证：

- 代码可正常导入。
- 关键配置可解析。
- 模型能完成最小前向。
- 数据集能正常加载。
- 主要训练、评测、推理或采样流程能跑通。
- 不明显破坏现有功能。

建议为新增内容补充以下测试之一或多项：

- dataset smoke test。
- model forward test。
- config load test。
- train/eval/predict/sample smoke test。
- 关键模块单测。
- 公共 builder、transform、collate、metric 或 scheduler 的回归测试。

模型复现类 PR 不强求一次性达到完全工程化，但至少要做到：

- 复现目标明确。
- 结果有基本支撑。
- README 可跑通。
- 配置与代码匹配。
- 主要指标可复现或可解释。

若仅完成“代码移植”而没有基本跑通与说明，不建议直接合入。

## 11. 代码风格与格式检查

本仓库使用 pre-commit 进行基础格式检查，当前包括 isort、black、ruff、Markdown CRLF/tab 检查、YAML/JSON 检查和 C/CUDA clang-format。

提交前建议执行：

```bash
pre-commit run --all-files
```

如新增或修改了测试，也建议执行对应测试命令，例如：

```bash
python -m pytest test
```

如果无法完整运行测试，请在 PR 描述中说明未运行项、原因和替代验证方式。

基本格式要求：

- Python 代码风格与现有仓库保持一致。
- YAML 和 JSON 格式合法。
- Markdown 不应包含 CRLF 和 tab。
- C/CUDA/C++/proto 代码需通过 clang-format。
- 不保留无关 debug 输出。
- 不提交冲突标记、私钥、token、无关临时文件。
- 不提交 `__pycache__`、`.pytest_cache`、训练输出、大日志或本地缓存。

以下内容通常不建议直接合入 PaddleMaterials 主仓库：

- 与仓库主任务方向无关的代码。
- 无 README、无说明、无最小验证的模型移植。
- 含个人本地绝对路径的代码。
- 强依赖私有环境的脚本。
- 临时实验文件。
- 大量未说明来源的权重或数据。
- 文档与代码明显不一致的实现。
- 只能在 Paddle develop 版本运行的功能。
- 完全绕开现有项目结构、另写一套训练框架的实现。

## 12. 提交前检查清单

提交 PR 前，请至少确认以下内容：

- 修改内容属于 PaddleMaterials 现有任务方向，或已与维护者讨论。
- 目录位置正确。
- 沿用了现有代码组织、配置方式和调用入口。
- 使用 PaddlePaddle 官方正式发布版本。
- 新增依赖合理，且版本和环境要求已说明。
- README 已补充且可用。
- 训练、评测、推理或采样主要流程能说明白。
- 配置文件能正常解析。
- 数据集或模型最小流程已验证。
- 已执行格式检查，或说明无法执行的原因。
- 未提交无关临时文件、大日志、缓存文件。
- 未提交无说明的大体积权重文件或数据文件。
- PR 描述完整，包含动机、修改内容、验证方式和已知限制。
- 如贡献属于指定活动，请按活动要求在对应 issue 或活动页同步进展，例如 `#194`。

## 13. Review 关注重点

Review 时，维护者通常会重点关注：

- 是否符合 PaddleMaterials 当前任务方向。
- 是否符合 repo 已有结构与逻辑。
- 是否复用了公共抽象，而不是绕开主流程。
- README 是否可支撑完整使用。
- 文档、配置和代码是否一致。
- 是否具备最小可复现性。
- 是否容易长期维护。
- 是否对现有功能造成副作用。
- 数据、权重和外部代码来源是否清晰合规。

请理解，review 的目标不是增加门槛，而是保证 PaddleMaterials 作为统一 AI4Materials 工具库的整体质量。

## 14. 致谢

感谢每一位贡献者对 PaddleMaterials 的支持。期待大家共同完善 PaddleMaterials 的模型、数据集、训练评测、推理采样和材料模拟能力。
