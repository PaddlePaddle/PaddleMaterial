# 高熵合金 CrewAI 多智能体优化系统

基于 **CrewAI** 框架的高熵合金 (HEA) 组分设计多智能体系统，在原有遗传算法框架（v1.3.0）基础上集成大语言模型智能体能力。

## 架构设计

系统遵循《面向高熵合金组分设计的主动学习型多智能体优化系统》知识库文档中的"**基于模型和效用的学习型智能体**"设计理念，将材料优化问题建模为序贯决策过程（POMDP），由六个专业智能体协同完成。

```
┌─────────────────────────────────────────────────────────────┐
│           高熵合金 CrewAI 多智能体系统 v1.0                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  用户自然语言需求                                            │
│       ↓                                                     │
│  ① Planner Agent        目标分析 → 搜索策略 → 元素池         │
│       ↓                                                     │
│  ② Knowledge Agent      历史记忆 + HEA 经验规则             │
│       ↓                                                     │
│  ③ Thermodynamic Agent  ΔS_mix / ΔH_mix / δ + CALPHAD      │
│       ↓                                                     │
│  ④ Property Prediction  VEC → 相结构 → 强度/延性预测         │
│       ↓                                                     │
│  ⑤ Optimization Agent   MEL 变异 + 进化搜索 + 采集函数       │
│       ↓                                                     │
│  ⑥ Memory Agent         持久化结果 → 生成报告                │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  LLM: LLMONE / OpenAI-compatible (base_url + model)        │
│  底层遗传引擎: AdaptiveEvolutionEngine (v1.3.0, 无LLM依赖)   │
└─────────────────────────────────────────────────────────────┘
```

## 智能体角色详解

| 智能体 | 文件 | 核心职责 | LLM 使用 |
|--------|------|---------|---------|
| **Planner Agent** | `agents/hea_crew.py` | 目标分解、策略制定 | ✅ |
| **Knowledge Agent** | `agents/hea_crew.py` | 历史记忆检索、HEA 规则 | ✅ |
| **Thermodynamic Agent** | `agents/hea_crew.py` | CALPHAD + 热力学评估 | ✅ |
| **Property Prediction Agent** | `agents/hea_crew.py` | 强度/延性预测 | ✅ |
| **Optimization Agent** | `agents/hea_crew.py` + `agents/strategy_agent.py` | MEL 变异、进化搜索 | ✅ |
| **Memory Agent** | `agents/hea_crew.py` | 结果存储、报告生成 | ✅ |

## 文件结构

```
ppmatAgent/hea_crewai_agent/
├── run_crewai.py                    # 主入口脚本
├── config.yaml                      # 遗传算法配置
│
├── agents/
│   ├── hea_crew.py                  # ★ CrewAI 多智能体实现（六大 Agent + Tools）
│   ├── llm_interface.py             # ★ LLMONE / OpenAI-compatible 调用接口
│   ├── strategy_agent.py            # MEL 变异 StrategyAgent（CrewAI LLM 版）
│   └── __init__.py
│
├── core/                            # 热力学计算核心（原 v1.3.0）
│   ├── material.py                  # Alloy 数据结构
│   ├── evaluator.py                 # ThermodynamicEvaluator
│   ├── calphad_evaluator.py         # CALPHADEvaluator (pycalphad)
│   ├── mel.py                       # MEL 操作语言
│   ├── evolution_engine.py          # 遗传算法引擎
│   └── tdb_registry.py              # TDB 数据库注册
│
├── tdb_files/                       # 热力学数据库文件
│   ├── mc_ni_v2036.tdb
│   ├── mc_fe_v2062.tdb
│   └── ...
│
├── examples/                        # 使用示例
│   ├── basic_evolution.py           # 纯遗传算法示例
│   └── calphad_evolution.py         # CALPHAD 进化示例
│
└── .ark/output/
    └── hea_memory.json              # 智能体长期记忆（自动维护）
```

## 安装

```bash
# 1. 安装 ppmatAgent 可选依赖
pip install -r ppmatAgent/requirements-optional.txt

# 2. 设置 API Key（默认 LLM API 是 llmone）
export LLMONE_API_KEY=your_llmone_api_key_here

# 3. (可选) 安装 CALPHAD 精确计算依赖
pip install pycalphad>=0.11.0
```

## 快速使用

### 命令行运行

```bash
# 基础运行（Co-Cr-Fe-Ni-V 体系）
python -m ppmatAgent.hea_crewai_agent.run_crewai

# 自定义需求
python -m ppmatAgent.hea_crewai_agent.run_crewai \
  --elements Co,Cr,Fe,Ni,V,Al \
  --requirement "设计一种抗氧化性强、800°C下高强度的六元高熵合金" \
  --output .ark/output/my_result/report.md

# 指定模型
python -m ppmatAgent.hea_crewai_agent.run_crewai --llm-api llmone --model gpt-5.2

# 使用任意 OpenAI-compatible 服务：DeepSeek、千帆、AI Studio 等都只改 base_url/model/key
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.deepseek.com/v1
python -m ppmatAgent.hea_crewai_agent.run_crewai --llm-api openai --model deepseek-chat

# AI Studio 示例
export AI_STUDIO_API_KEY=...
python -m ppmatAgent.hea_crewai_agent.run_crewai \
  --llm-api openai \
  --base-url https://aistudio.baidu.com/llm/lmapi/v3 \
  --model ernie-5.0-thinking-preview
```

### Python API

```python
from ppmatAgent.hea_crewai_agent.agents.hea_crew import HEACrewOptimizer

optimizer = HEACrewOptimizer(
    element_pool=["Co", "Cr", "Fe", "Ni", "V"],
    user_requirement="设计一种在800°C下屈服强度>900MPa的高熵合金",
    model="gpt-5.2",
    llm_api="llmone",
)
report = optimizer.run()
print(report)
```

### LLM 配置参考

HEA CrewAI 只区分两类接口：`llmone` 和 `openai`。`openai` 表示
OpenAI-compatible API，DeepSeek、千帆、AI Studio 等都通过 `base_url + model`
切换，不再作为独立 provider。

LLMONE（默认）：

```bash
HEA_LLM_API=llmone
LLMONE_API_KEY=...
LLMONE_BASE_URL=https://oneapi-comate.baidu-int.com/v1
```

OpenAI-compatible：

```bash
HEA_LLM_API=openai
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
```

DeepSeek 示例：

```bash
HEA_LLM_API=openai
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.deepseek.com/v1
```

AI Studio 示例：

```bash
HEA_LLM_API=openai
AI_STUDIO_API_KEY=...
OPENAI_BASE_URL=https://aistudio.baidu.com/llm/lmapi/v3
```

内部会统一构造 CrewAI `LLM`：

```python
from crewai import LLM

llm = LLM(
    model="ernie-5.0-thinking-preview",
    temperature=0.7,
    api_key=os.getenv("AI_STUDIO_API_KEY", ""),
    base_url="https://aistudio.baidu.com/llm/lmapi/v3",
)
```

## 工作流程

```
用户自然语言输入
      │
      ▼ (Task 1)
  Planner: 分析目标 → 元素池 + 权重 + 策略 JSON
      │
      ▼ (Task 2)
  Knowledge: 历史记忆检索 + HEA 经验规则摘要
      │
      ▼ (Task 3)
  Thermodynamic: 生成3个候选成分 + 热力学评估
                 ├─ thermodynamic_evaluator (ΔS, ΔH, δ)
                 └─ calphad_evaluator (相平衡，如可用)
      │
      ▼ (Task 4)
  Property Prediction: VEC分析 + 力学性能预测
      │
      ▼ (Task 5)
  Optimization: GraphRAG 引导 MEL 变异 + 真实进化搜索
                ├─ evolution_search_runner (AdaptiveEvolutionEngine)
                └─ mel_mutation_generator (LLM + GraphRAG，可降级为 GraphRAG/规则)
      │
      ▼ (Task 6)
  Memory: 保存高价值成分 + 生成 Markdown 报告
          └─ .ark/output/hea_memory.json
```

## 内置工具

| 工具 | 用途 |
|------|------|
| `thermodynamic_evaluator` | 快速热力学评估（ΔS_mix, ΔH_mix, δ） |
| `calphad_evaluator` | CALPHAD 相平衡精确计算 |
| `mel_mutation_generator` | LLM + GraphRAG 驱动的 MEL 变异操作生成 |
| `memory_reader` | 历史搜索结果检索 |
| `memory_writer` | 高价值成分持久化存储 |

## 与原版本对比

| 特性 | v1.3.0（原始） | v1.0 CrewAI（本版本） |
|------|--------------|---------------------|
| 优化算法 | 遗传算法（纯数学） | 遗传算法 + LLM 智能体协同 |
| MEL 变异 | 随机规则生成 | LLM/GraphRAG 目标导向生成 |

### 组分优化 GraphRAG

`StrategyAgent` 会在每次生成 MEL 变异前查询本地 `materials_graphrag`：

- 从 `.ark/output/hea_memory.json` 读取历史高分/低分候选；
- 检索元素、性能、候选成分和代理模型节点；
- 将 GraphRAG 摘要和证据路径注入 MEL prompt；
- 当 `strategy_agent.enabled=false` 时，不调用 LLM，但仍使用 GraphRAG + 规则生成变异。

配置：

```yaml
strategy_agent:
  enabled: false
  graphrag:
    enabled: true
    max_results: 5
```
| 目标输入 | 手动配置权重 | 自然语言描述 |
| 结果报告 | 模板填充 | LLM 生成专业分析报告 |
| 知识积累 | 无 | 智能体长期记忆（JSON） |
| 智能体数量 | 0 | 6 个专业智能体 |

## 性能说明

- **离线计算**（热力学评估）：1-2 分钟完成
- **LLM API 调用**：约 20-50 次（取决于 Crew 迭代深度）
- **总耗时预估**：5-15 分钟（含 API 延迟）
