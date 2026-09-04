# LiFlow 最终验收报告

更新日期：2026-09-04

## 1. 验收范围

本报告汇总当前 PaddleMaterials `develop` 基线上的 LiFlow 复现结果，覆盖模型、数据、权重、训练、推理、GPU eager 基准和已知限制。

冻结版本：

- LiFlow reference：`e6fc475361d046865f12cae1aee11c4f56c48d87`
- PaddleMaterials：`a9a689cf64d98a2b415a15f794fe88598b5943ef`
- Python 环境：`ppmat-liflow`，Python 3.10
- GPU：NVIDIA GeForce RTX 5060，driver 610.47
- Paddle：`paddlepaddle-gpu 3.3.1`

## 2. 验收矩阵

| 验收项 | 结果 | 证据/说明 |
|---|---|---|
| LiFlow layers 与网络移植 | 通过 | `test_liflow_layers.py`、`test_liflow_model.py` |
| Forward 数值对齐 | 通过 | 三个 fixture，最大误差 `1.239e-7`，阈值 `1e-6` |
| Backward 梯度对齐 | 通过 | 三个 fixture，最大误差 `1.192e-7`，阈值 `1e-6` |
| 两阶段训练与 checkpoint continuation | 通过 | `test_liflow_align.py` 与 `test_liflow_integration.py`，此前结果 `13 passed` |
| Universal 真实数据 schema/周期图 | 通过 | `test_liflow_real_assets.py`，结果 `2 passed` |
| Universal 真实 P/C checkpoint 前向与短轨迹 | 通过 | `test_liflow_real_assets.py`，结果 `2 passed` |
| LGPS 真实数据与 P/C checkpoint | 通过 | `test_liflow_lgps_lps.py`，结果 `4 passed` |
| LPS 端到端数据验证 | 有意跳过 | 用户明确要求跳过；本地没有官方 LPS 数据集 |
| GPU eager 推理 | 通过 | RTX 5060，真实 Universal 样本 `mp_4426`，160 atoms |
| CINN 后端 | 阻塞 | 当前 Windows GPU wheel `cinn_build=False` |
| eager/CINN 加速比 `>30%` | 未完成 | 无 CINN 编译支持，不能计算或宣称加速 |
| 采样 MSD/RDF/final-step 误差 `<=5%` | 未完成 | 尚未取得可用于 PyTorch/Paddle 成对比较的完整评估结果 |
| 模型/数据公开下载包 | 部分完成 | 本地 POSIX 模型 archive 已 round-trip 验证；尚无稳定公开 BCE URL |
| 清洁环境安装与完整 CI/CE | 未完成 | 尚未形成可复核的清洁环境和完整 CI/CE 证据 |

## 3. GPU eager 基准

命令使用真实 Universal 数据和转换后的 Propagator：

```powershell
$env:PATH = "C:\Users\YU\miniconda3\envs\ppmat-liflow\Lib\site-packages\nvidia\cu13\bin\x86_64;C:\Users\YU\miniconda3\envs\ppmat-liflow\Library\bin;$env:PATH"
C:\Users\YU\miniconda3\envs\ppmat-liflow\python.exe molecular_dynamics_integrator\liflow\benchmark.py --backend both --data-root "D:\实验\liflow_official_data\data\universal" --checkpoint "D:\实验\liflow_reference\ckpt\converted\P_universal\propagator.pdparams" --warmup 20 --iterations 100 --output molecular_dynamics_integrator\liflow\benchmark_gpu.json
```

结果：

- 首次调用：`1.287954 s`
- 20 次 warmup 平均：`5.5288 ms`
- 100 次正式迭代平均：`6.0974 ms`
- 正式迭代标准差：`1.6382 ms`
- 设备：`gpu:0`
- CUDA 编译支持：`True`
- CINN 编译支持：`False`

原始结果见 [benchmark_gpu.json](../benchmark_gpu.json)。CINN 的跳过原因为：

```text
RuntimeError: LiFlow execution_backend='cinn' requires a Paddle build with CINN support.
```

## 4. 权重转换审计

Universal Propagator 和 Corrector 均完成转换和回载审计：

- 每个 checkpoint 45 个 state-dict entries；
- 每个 checkpoint 加载 45 个参数；
- 每个 checkpoint 22 个 Linear 权重执行显式转置；
- `missing_keys=[]`；
- `unexpected_keys=[]`；
- `shape_mismatch=[]`；
- reload-back check：clean。

详细记录见 [alignment_report.md](alignment_report.md)。

## 5. 当前最终结论

当前实现已通过模型层、完整模型前向/反向、两阶段训练、checkpoint continuation、Universal/LGPS 真实资产和 GPU eager 推理验证。

项目尚未达到全部验收标准，剩余阻塞项为：

1. 当前官方 Windows GPU Paddle wheel 没有 CINN 编译支持，无法执行 CINN 性能对比；
2. 尚未完成 PyTorch/Paddle 真实采样的 MSD、RDF 和 final-step `<=5%` 对比；
3. 尚未完成稳定公开模型/数据包、清洁环境安装和完整 CI/CE 的最终证据；
4. LPS 端到端验证按用户要求跳过，不作为失败替代或数据替代。

因此最终状态为：**核心 LiFlow 复现通过；完整验收部分通过，Task 17 CINN、采样指标和发布/清洁环境验收仍未闭环。**
