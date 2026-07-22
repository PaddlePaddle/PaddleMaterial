#!/usr/bin/env python3
"""
高熵合金 CrewAI 多智能体优化器 - 快速入口
示例: python -m ppmatAgent.hea_crewai_agent.run_crewai --elements Co,Cr,Fe,Ni,V --requirement "高强韧性合金"
"""

import os
import sys
from pathlib import Path

# 确保项目根目录在 PYTHONPATH
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from ppmatAgent.hea_crewai_agent.runtime_config import (
    describe_dependency_status,
    get_crewai_defaults,
    load_env_file,
    load_runtime_config,
    resolve_env_path,
    resolve_config_path,
)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="高熵合金 CrewAI 多智能体优化系统"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML 配置文件路径（默认: 项目根目录下的 config.yaml）",
    )
    parser.add_argument(
        "--elements",
        default=None,
        help="元素池，逗号分隔（默认读取 config.yaml 或使用内置默认值）",
    )
    parser.add_argument(
        "--requirement",
        default=None,
        help="自然语言优化需求（默认读取 config.yaml 或使用内置默认值）",
    )
    parser.add_argument(
        "--llm-api",
        default=None,
        choices=["llmone", "openai"],
        help="LLM API type: llmone or openai-compatible（默认读取 config.yaml 或 HEA_LLM_API）",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["llmone", "openai", "openai-compatible", "deepseek", "qianfan", "wenxin-aistudio"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL override. Also configurable with HEA_LLM_BASE_URL / OPENAI_BASE_URL / LLMONE_BASE_URL.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM 模型名称（默认读取 config.yaml 或使用内置默认值）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="报告输出路径（默认读取 config.yaml 或使用内置默认值）",
    )
    args = parser.parse_args()

    load_env_file()
    config_path = resolve_config_path(args.config)
    env_path = resolve_env_path()
    try:
        runtime_config = load_runtime_config(config_path)
    except Exception as exc:
        print(f"[错误] 无法加载配置文件 {config_path}: {exc}")
        print("  请检查 PyYAML 依赖和 config.yaml 格式。")
        return 1
    crewai_defaults = get_crewai_defaults(runtime_config)
    os.environ["HEA_CONFIG_PATH"] = str(config_path)

    elements_value = args.elements or ",".join(crewai_defaults["default_elements"])
    requirement_value = args.requirement or crewai_defaults["default_requirement"]
    llm_api_value = (
        args.llm_api
        or args.provider
        or os.getenv("HEA_LLM_API")
        or os.getenv("HEA_LLM_PROVIDER")
        or crewai_defaults["default_llm_api"]
    )
    model_value = args.model or crewai_defaults["default_model"]
    output_value = args.output or crewai_defaults["output_report"]

    dependency_status = describe_dependency_status()
    if dependency_status["missing_optional"]:
        print("[提示] 可选依赖缺失:", ", ".join(dependency_status["missing_optional"]))
        print("  - pycalphad: 精确 CALPHAD 相平衡计算")
        print("  - ase: MD/ORCA 结构生成与高精度验证")

    if dependency_status["missing_required"]:
        print("[错误] 缺少运行 run_crewai.py 所需依赖:", ", ".join(dependency_status["missing_required"]))
        print("  请先执行: pip install -r ppmatAgent/requirements-optional.txt")
        return 1

    from ppmatAgent.hea_crewai_agent.agents.hea_crew import HEACrewOptimizer

    element_pool = [e.strip() for e in elements_value.split(",") if e.strip()]
    output_path = ROOT / output_value

    print("=" * 60)
    print("  高熵合金 CrewAI 多智能体优化系统 v1.0")
    print("=" * 60)
    print(f"  元素池     : {element_pool}")
    print(f"  优化需求   : {requirement_value[:60]}...")
    print(f"  LLM API     : {llm_api_value}")
    if args.base_url:
        print(f"  Base URL    : {args.base_url}")
    print(f"  使用模型   : {model_value}")
    print(f"  配置文件   : {config_path}")
    print(f"  环境文件   : {env_path}")
    print(f"  输出报告   : {output_path}")
    print("=" * 60)

    optimizer = HEACrewOptimizer(
        element_pool=element_pool,
        user_requirement=requirement_value,
        model=model_value,
        llm_api=llm_api_value,
        base_url=args.base_url,
        temperature=0.7,
        config_path=str(config_path),
    )

    report = optimizer.run()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print("\n" + "=" * 60)
    print("优化完成！最终报告（摘要）：")
    print("=" * 60)
    print(report[:1000] + ("..." if len(report) > 1000 else ""))
    if optimizer.last_memory_sync_messages:
        print("\nMemory 同步：")
        for message in optimizer.last_memory_sync_messages:
            print(f"  - {message}")
    print(f"\n完整报告已保存至: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
