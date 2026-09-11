"""Transcriber selection: tsumugi (default) vs MuScriptor."""

from __future__ import annotations

import pytest

from sound2midi import muscriptor
from sound2midi.amt import MODEL_TYPES, default_amt_home, default_analysis_home
from sound2midi.cli import _build_parser, _resolve_model


def _args(argv):
    parser = _build_parser()
    return parser, parser.parse_args(argv)


def test_default_transcriber_and_model():
    parser, args = _args(["song.wav"])
    assert args.transcriber == "tsumugi"
    assert _resolve_model(args, parser) == "default"


def test_tsumugi_accepts_the_new_upstream_variants():
    for variant in ("bass_v2", "guitar_v1_5", "drums_v1_5", "other_v1_5", "vocal_harmony_v1_5"):
        assert variant in MODEL_TYPES
        parser, args = _args(["song.wav", "--type", variant])
        assert _resolve_model(args, parser) == variant


def test_muscriptor_sizes_and_passthrough():
    for size in muscriptor.MODEL_SIZES:
        parser, args = _args(["song.wav", "--transcriber", "muscriptor", "--type", size])
        assert _resolve_model(args, parser) == size
    parser, args = _args(["song.wav", "--transcriber", "muscriptor"])
    assert _resolve_model(args, parser) == "medium"  # upstream's default
    # paths and hf:// URLs are handed through untouched
    for custom in ("hf://MuScriptor/muscriptor-large", "/tmp/weights.safetensors"):
        parser, args = _args(["song.wav", "--transcriber", "muscriptor", "--type", custom])
        assert _resolve_model(args, parser) == custom


def test_models_are_validated_per_transcriber():
    parser, args = _args(["song.wav", "--type", "large"])  # a MuScriptor size
    with pytest.raises(SystemExit):
        _resolve_model(args, parser)
    parser, args = _args(["song.wav", "--transcriber", "muscriptor", "--type", "guitar_v1_5"])
    with pytest.raises(SystemExit):
        _resolve_model(args, parser)


def test_environments_are_separate():
    """skey pins torch<2.8 and tsumugi pins 2.13, so they must not share a venv."""
    homes = {default_amt_home(), default_analysis_home(), muscriptor.default_muscriptor_home()}
    assert len(homes) == 3
    assert default_amt_home().name == "tsumugi"


def test_muscriptor_rejects_unknown_formats(tmp_path):
    with pytest.raises(ValueError):
        muscriptor.transcribe(
            tmp_path / "a.wav", tmp_path / "a.mid", home=tmp_path, output_format="bogus"
        )


# --- per-model output folders (comparing runs side by side) ------------------


def test_model_tag_per_transcriber_and_mode():
    from sound2midi.cli import _model_tag

    parser, args = _args(["song.wav"])
    assert _model_tag(args, _resolve_model(args, parser)) == "tsumugi-default"

    parser, args = _args(["song.wav", "--type", "other_v1_5"])
    assert _model_tag(args, _resolve_model(args, parser)) == "tsumugi-other_v1_5"

    parser, args = _args(["song.wav", "--transcriber", "muscriptor", "--type", "large"])
    assert _model_tag(args, _resolve_model(args, parser)) == "muscriptor-large"

    # stems pick a model per stem, so the mode names the folder
    parser, args = _args(["song.wav", "--stems"])
    assert _model_tag(args, _resolve_model(args, parser)) == "tsumugi-stems"


def test_model_tags_are_distinct_so_runs_do_not_overwrite():
    from sound2midi.cli import _model_tag

    runs = [
        [],
        ["--type", "other_v1_5"],
        ["--stems"],
        ["--transcriber", "muscriptor"],
        ["--transcriber", "muscriptor", "--type", "large"],
    ]
    tags = set()
    for extra in runs:
        parser, args = _args(["song.wav", *extra])
        tags.add(_model_tag(args, _resolve_model(args, parser)))
    assert len(tags) == len(runs)
