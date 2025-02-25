import argparse
import sys
from typing import Callable
from typing import TypeVar
from typing import cast

from omegaconf import OmegaConf

R = TypeVar("R")


def get_config(argv: (list[str] | None), config_cls: Callable[..., R]) -> R:
    """
    Utility function to get OmegaConf config options.

    Args:
        argv: Either a list of command line arguments to parse, or None.
            If None, this argument is set from sys.argv.
        config_cls: Dataclass object specifying config structure
            (i.e. which fields to expect in the config).
            It should be the class itself, NOT an instance of the class.

    Returns:
        Config object, which will pass as an instance of `config_cls` among other things.
            Note: the type for this could be specified more carefully, but OmegaConf's typing
            system is a bit complex. See OmegaConf's docs for "structured" for more info.
    """
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config",
        type=str,
        action="append",
        default=list(),
        help="Path to a yaml config file. Argument can be repeated multiple times, with later configs overwriting previous ones.",
    )
    args, config_changes = parser.parse_known_args(argv)
    conf_yamls = [OmegaConf.load(c) for c in args.config]
    conf_cli = OmegaConf.from_cli(config_changes)
    schema = OmegaConf.structured(config_cls)
    config = OmegaConf.merge(schema, *conf_yamls, conf_cli)
    OmegaConf.set_readonly(config, True)
    return cast(R, config)
