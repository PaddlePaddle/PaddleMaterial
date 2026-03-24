# MatInvent 对齐验证

本目录包含 MatInvent 权重迁移的完整验证脚本，覆盖四个组件（diffcsp / mattergen / rl / matinvent），
分别验证前向推理精度、训练 loss 对齐、采样指标三项内容。

注意：因为 MatInvent 原版代码已经把噪声发生器的一些逻辑完全写死在主框架逻辑中，导致没办法直接让 paddle这边的移植逻辑直接和
原版对齐。 所以在test 测试这里加了 monkey patch的方式，算是强行劫持了噪声发生器。仅供在单元测试对齐时临时性的patch测试

## 性能说明

因为原版默认1000步采样的设定，在对齐测试的时候，根据当前电脑性能的不同，可能需要几分钟甚至几十分钟才能结束的情况。

执行过程中长时间没跑完是正常情况，耐心等待结束。

## 文件说明

```
matinvent_test.py               主验证脚本，包含全部 step_1 ~ step_24
pt_runner_forward_diffcsp.py    DiffCSP前向对齐 PyTorch 子进程脚本
pt_runner_forward_mattergen.py  MatterGen前向对齐 PyTorch 子进程脚本
pt_runner_training_diffcsp.py   DiffCSP 训练 PyTorch 子进程脚本
pt_runner_training_mattergen.py MatterGen 训练 PyTorch 子进程脚本
pt_runner_sampling_diffcsp.py   DiffCSP 采样 PyTorch 子进程脚本
pt_runner_sampling_mattergen.py MatterGen 采样 PyTorch 子进程脚本
```

pt_runner_* 脚本由 matinvent_test.py 通过 subprocess 在 matinvent conda 环境下调用，
不需要手动执行，直接手动强制执行会报错。

## 关键参数说明

在日志中你会看到类似命令：

```bash
... pt_runner_sampling_diffcsp.py <raw-matinvent路径> <output_json> <seed> <num_samples> <num_atoms_list> <num_steps>
```

其中第一个业务参数 `<raw-matinvent路径>` 是必须的，作用如下：

1. 传给 `pt_runner_*` 子脚本作为 `RAW_ROOT`。
2. 子脚本会执行 `sys.path.insert(0, RAW_ROOT)`。
3. 这样才能从 `raw-matinvent` 导入原版模块（例如 `models.diffcsp.*`、`mattergen.*`）用于 PyTorch 侧交叉验证。

如果缺少这个参数或路径不正确，常见报错是：

- `ModuleNotFoundError: No module named 'models.diffcsp.diffusion_utils'`

## 运行前置条件

1. 环境：ppmat conda 环境用于运行主脚本；matinvent conda 环境需要单独安装（供 pt_runner 子进程使用，需要确保
   matinvent的环境里可以正常运行原始 matinvent 的程序，不然无法直接推理采样进行交叉验证）。

2. Paddle 权重：脚本首次运行时会自动通过 HTTP 下载并存到 `~/.paddlemat/weights/matinvent/`，无需手动操作。

   也可以手动下载：
   ```bash
   # 方式一：下载到本地后放到 ~/.paddlemat/weights/matinvent/
   wget https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/MatInvert/matinvent_diffcsp_mp20.pdparams
   wget https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/MatInvert/matinvent_mattergen_mp20.pdparams
   ```
   或在 AI Studio 上可下载：https://aistudio.baidu.com/modelsdetail/44805/space

3. PyTorch 原版权重（matinvent自己组网训练的权重，用来做交叉对比）：脚本首次运行时会自动通过 HuggingFace API 下载，也可以手动下载后备份到
   `~/.paddlemat/weights/matinvent/`：
    - `raw-diffcsp.ckpt`（来自 https://huggingface.co/jwchen25/MatInvent/diffcsp_mp20/last.ckpt）
    - `raw-mattergen.ckpt`（来自 https://huggingface.co/jwchen25/MatInvent/mattergen_base/last.ckpt）

特别注意，diffcsp 和 mattergen 这2块权重，都是 matinvent自己自定义组网训练的，不要直接从这2个框架的官方里下载权重使用，原版程序会参数不匹配报错

下载速度慢时，可配置镜像：

   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

## 运行方法

在项目根目录下执行：

```bash
conda activate ppmat
python matinvent_test.py
```

或者用绝对路径在任意目录执行：

```bash
/home/cao/miniconda3/envs/ppmat/bin/python matinvent_test.py
```

### 只验证单个模型

```bash
python matinvent_test.py --model diffcsp
python matinvent_test.py --model mattergen
python matinvent_test.py --model rl
python matinvent_test.py --model matinvent
```

### 只跑某类检查

```bash
python matinvent_test.py --check forward
python matinvent_test.py --check training
python matinvent_test.py --check sampling
```

## 结果说明

脚本结束时会在终端打印一张汇总表，形如：

```
========================================================================
模型            前向Logits      训练对齐        采样指标        整体
========================================================================
diffcsp         PASS            PASS            PASS            PASS
mattergen       PASS            PASS            PASS            PASS
rl              PASS            PASS            PASS            PASS
matinvent       PASS            PASS            PASS            PASS
========================================================================
```

JSON 和 MD 格式的详细报告写入 `/tmp/` 下。

**退出码**：有任何 不PASS 时退出码为 1，全部 PASS 时退出码为 0。

## 环境变量设置

| 变量                         | 默认值   | 说明                                              |
|----------------------------|-------|-------------------------------------------------|
| `MATINVENT_PT_TIMEOUT_SEC` | 600   | PyTorch 子进程超时时间（秒）                              |
| `HF_ENDPOINT`              | （未设置） | HuggingFace 镜像，建议国内用户设为 `https://hf-mirror.com` |

## 常见问题

**PyTorch 子进程报 ModuleNotFoundError**：检查 matinvent conda 环境是否安装了 mattergen / diffcsp 依赖

**Paddle 权重下载失败**：检查网络连通性，或者参考上方「前置条件」手动下载并放到 `~/.paddlemat/weights/matinvent/`。
