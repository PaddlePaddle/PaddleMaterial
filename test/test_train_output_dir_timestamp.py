import datetime
import re
from pathlib import Path

from omegaconf import OmegaConf

from ppmat.utils.output_dir import append_timestamp_to_output_dir


def test_append_timestamp_to_output_dir_uses_seed_and_timestamp():
    config = OmegaConf.create({"Trainer": {"output_dir": "./output/demo", "seed": 7}})

    append_timestamp_to_output_dir(
        config,
        now=datetime.datetime(2026, 6, 29, 12, 34, 56),
    )

    assert config["Trainer"]["output_dir"] == "./output/demo_t_20260629_123456_s_7"


def test_train_entrypoints_append_timestamp_by_default():
    train_scripts = [
        "property_prediction/train.py",
        "electronic_structure/train.py",
        "structure_generation/train.py",
        "spectrum_elucidation/train.py",
        "interatomic_potentials/train.py",
        "spectrum_enhancement/train.py",
    ]

    for script in train_scripts:
        source = Path(script).read_text()
        assert re.search(r"append_timestamp_to_output_dir\((config|cfg)\)", source)
