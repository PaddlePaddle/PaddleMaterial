from pathlib import Path

import pytest


def test_knowmat_import_does_not_require_llm_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    import ppmatAgent.knowmat
    import ppmatAgent.knowmat.config as config

    assert callable(ppmatAgent.knowmat.run)
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        config.ensure_runtime_env(require_llm=True)


def test_knowmat_package_resources_load():
    from ppmatAgent.knowmat.domain_rules import DomainRules, default_rules
    from ppmatAgent.knowmat.prompt_loader import load_text_template

    assert default_rules.valid_elements
    assert "Powder_Metallurgy" in default_rules.process_category_keywords
    assert "Beam_Current_mA" in default_rules.parameter_patterns
    assert isinstance(DomainRules.from_yaml().phase_patterns, dict)
    assert load_text_template("extraction_system_template.txt").strip()


def test_domain_rules_from_custom_yaml(tmp_path: Path):
    from ppmatAgent.knowmat.domain_rules import DomainRules

    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        """
valid_elements:
  - Ti
  - Nb
phase_patterns:
  bcc: BCC
precipitate_keywords:
  - sigma
property_name_mapping:
  hardness: Hardness
process_category_keywords:
  AM_DED:
    - directed energy deposition
parameter_patterns:
  Laser_Power_W:
    - "(\\\\d+\\\\.\\\\d+) W"
""",
        encoding="utf-8",
    )

    rules = DomainRules.from_yaml(rules_path)

    assert rules.valid_elements == {"Ti", "Nb"}
    assert rules.phase_patterns["bcc"] == "BCC"
    assert "Laser_Power_W" in rules._compiled_param_patterns


def test_validate_dotenv_file_quotes(tmp_path: Path):
    from ppmatAgent.knowmat.env_loader import validate_dotenv_file

    valid_env = tmp_path / ".env.valid"
    valid_env.write_text('LLM_MODEL="deepseek-chat"\nOCR_RENDER_DPI="300"\n', encoding="utf-8")
    validate_dotenv_file(str(valid_env))

    invalid_env = tmp_path / ".env.invalid"
    invalid_env.write_text('LLM_MODEL="deepseek-chat"\nOCR_BATCH_SIZE="4\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"line 2: missing closing \" quote"):
        validate_dotenv_file(str(invalid_env))


def test_hea_tdb_registry_uses_packaged_files():
    from ppmatAgent.hea_crewai_agent.core.tdb_registry import (
        find_best_local_tdb,
        get_local_tdb_registry,
    )

    registry = get_local_tdb_registry()
    assert any(entry["exists"] for entry in registry)

    selected = find_best_local_tdb(["Co", "Cr", "Fe", "Ni", "V"])
    assert selected is not None
    assert selected["filename"] == "CoCrFeNiV.TDB-R3.txt"
    assert Path(selected["path"]).is_file()


def test_surrogate_registry_prefers_paddlematerials():
    from ppmatAgent.hea_crewai_agent.core.surrogate_models import (
        detect_surrogate_backends,
        list_surrogate_model_candidates,
        recommend_surrogate_stack,
    )

    recommendation = recommend_surrogate_stack(
        "high entropy alloy property prediction and interatomic potential",
        prefer_paddle=True,
    )
    candidates = list_surrogate_model_candidates("composition property prediction")
    backends = detect_surrogate_backends()

    assert recommendation["recommended"][0]["id"] == "paddlematerials"
    assert any(item["id"] == "crabnet" for item in candidates)
    assert "paddlematerials" in backends
    assert "installed" in backends["paddlematerials"]
