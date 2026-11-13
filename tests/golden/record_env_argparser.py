"""Record every argument of dexmachina's common env argparser into tests/golden/env_argparser.json.

The fixture pins option strings, defaults, types and order, so that tests/test_env_spec.py can show the parser
generated from EnvSpec is identical to the one it replaced. Record deliberately, on purpose-made changes only:
    uv run --exact --extra flashsac python tests/golden/record_env_argparser.py
"""

import argparse
import json
from pathlib import Path
from typing import Any

SNAPSHOT_FILE = Path(__file__).with_name("env_argparser.json")


def describe_parser(parser: argparse.ArgumentParser) -> dict[str, Any]:
    """JSON-comparable description of every action except help, in the order the parser holds them."""
    actions = [action for action in parser._actions if not isinstance(action, argparse._HelpAction)]
    return {
        "order": [action.dest for action in actions],
        "actions": [
            {
                "option_strings": action.option_strings,
                "dest": action.dest,
                "default": action.default,
                "type": None if action.type is None else action.type.__name__,
                "choices": None if action.choices is None else list(action.choices),
                "nargs": action.nargs,
                "const": action.const,
                "required": action.required,
                "help": action.help,
                "action": type(action).__name__,
            }
            for action in actions
        ],
    }


if __name__ == "__main__":
    from dexmachina.envs.constructors import get_common_argparser

    SNAPSHOT_FILE.write_text(json.dumps(describe_parser(get_common_argparser()), indent=2) + "\n")
    print(f"wrote {SNAPSHOT_FILE}")
