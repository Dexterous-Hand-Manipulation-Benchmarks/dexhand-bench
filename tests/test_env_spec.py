"""EnvSpec is the one source of the env settings; the common argparser generated from it must equal the old one."""

import dataclasses
import json
from typing import Any

import pytest

from conftest import GOLDEN_DIR
from golden.record_env_argparser import describe_parser

constructors = pytest.importorskip("dexmachina.envs.constructors", reason="needs genesis and dexmachina")
from dexmachina.envs.spec import EnvSpec  # noqa: E402  (after the skip: needs dexmachina)

NON_DEFAULT_ARGV = [
    "-B", "8",  # int
    "--first_ratio", "0.25",  # float
    "--curr_schedule", "exp",  # str with choices
    "--vis",  # store_true
    "--hybrid_scales", "0.1", "0.2", "0.3",  # nargs list
    "--color_object", "red",  # None default
]  # fmt: skip


@pytest.fixture(scope="module")
def recorded() -> dict[str, Any]:
    return json.loads((GOLDEN_DIR / "env_argparser.json").read_text())


def test_parser_matches_recorded_snapshot(recorded: dict[str, Any]) -> None:
    current = describe_parser(constructors.get_common_argparser())
    assert current["order"] == recorded["order"]
    for now, then in zip(current["actions"], recorded["actions"], strict=True):
        assert now == then


def test_parse_args_defaults_match_recorded_snapshot(recorded: dict[str, Any]) -> None:
    parsed = vars(constructors.get_common_argparser().parse_args([]))
    assert parsed == {action["dest"]: action["default"] for action in recorded["actions"]}
    for action in recorded["actions"]:
        value, default = parsed[action["dest"]], action["default"]
        assert type(value) is type(default), action["dest"]
        if isinstance(default, list):
            assert [type(x) for x in value] == [type(x) for x in default], action["dest"]


def test_parsed_defaults_give_the_default_spec() -> None:
    assert EnvSpec.from_args(constructors.get_common_argparser().parse_args([])) == EnvSpec()


def test_args_round_trip_for_non_default_argv() -> None:
    args = constructors.get_common_argparser().parse_args(NON_DEFAULT_ARGV)
    spec = EnvSpec.from_args(args)
    assert spec.hybrid_scales == (0.1, 0.2, 0.3)
    assert vars(spec.to_args()) == vars(args)
    assert isinstance(spec.to_args().hybrid_scales, list)


def test_from_args_ignores_attributes_added_by_methods() -> None:
    args = constructors.get_common_argparser().parse_args([])
    args.method_specific_setting = 1
    assert EnvSpec.from_args(args) == EnvSpec()


def test_spec_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        EnvSpec().num_envs = 2  # type: ignore[misc]
