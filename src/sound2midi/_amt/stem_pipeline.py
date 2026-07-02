"""Stem-separated transcription, replicating the upstream Colab notebook.

This script is NOT part of the importable ``sound2midi`` package and is not linted
or type-checked with the rest of the project. It is executed by the *AMT* virtualenv
(via ``sound2midi.amt.transcribe_stems``) because it imports the AMT repo's ``infer``
module and the ``stem_splitter`` package, neither of which live in this project's env.

Flow (faithful to ``Colab_Inference.ipynb``):
  1. Separate the input audio into stems with stem-splitter (BS-RoFormer).
  2. Transcribe each stem with the AMT model matching the stem name.
  3. Merge the per-stem MIDIs into one file.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path


def _bootstrap_amt_repo() -> None:
    """Put the AMT repo on sys.path so ``import infer`` resolves."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--amt-repo", required=True)
    known, _ = parser.parse_known_args()
    sys.path.insert(0, known.amt_repo)


_bootstrap_amt_repo()

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import pretty_midi  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

import infer  # noqa: E402  (from the AMT repo)
from stem_splitter.inference import (  # noqa: E402
    SeparationConfig,
    _separate_one_file,
    load_mss_model,
)

STEM_PIPELINE_CACHE: dict = {}


def merge_midis_logic(midi_paths, output_file, max_melodic=15):
    """Merge per-stem MIDIs into one, aggregating by (program, is_drum, name)."""
    if not midi_paths:
        raise ValueError("No MIDI files to merge")

    # Use the first MIDI's tempo map as the master timeline.
    master_pm = pretty_midi.PrettyMIDI(str(midi_paths[0]))

    all_notes = defaultdict(list)
    all_ccs = defaultdict(list)
    all_pbends = defaultdict(list)
    instrument_names = {}

    for path in midi_paths:
        pm = pretty_midi.PrettyMIDI(str(path))
        for inst in pm.instruments:
            key = (inst.program, inst.is_drum, inst.name)
            filtered_notes = [n for n in inst.notes if (n.end - n.start) < 15.0]
            all_notes[key].extend(filtered_notes)
            all_ccs[key].extend(inst.control_changes)
            all_pbends[key].extend(inst.pitch_bends)
            if key not in instrument_names:
                instrument_names[key] = inst.name

    melodic_keys = [k for k in all_notes if not k[1]]
    drum_keys = [k for k in all_notes if k[1]]
    melodic_keys.sort(key=lambda k: len(all_notes[k]), reverse=True)

    final_instruments = []
    if len(melodic_keys) > max_melodic:
        kept_keys = melodic_keys[: max_melodic - 1]
        overflow_keys = melodic_keys[max_melodic - 1 :]
        for key in kept_keys:
            inst = pretty_midi.Instrument(
                program=key[0], is_drum=key[1], name=instrument_names[key]
            )
            inst.notes = all_notes[key]
            inst.control_changes = all_ccs[key]
            inst.pitch_bends = all_pbends[key]
            final_instruments.append(inst)

        base_key = overflow_keys[0]
        overflow_inst = pretty_midi.Instrument(
            program=base_key[0], is_drum=base_key[1], name="Other / Merged"
        )
        for key in overflow_keys:
            overflow_inst.notes.extend(all_notes[key])
            overflow_inst.control_changes.extend(all_ccs[key])
            overflow_inst.pitch_bends.extend(all_pbends[key])
        final_instruments.append(overflow_inst)
    else:
        for key in melodic_keys:
            inst = pretty_midi.Instrument(
                program=key[0], is_drum=key[1], name=instrument_names[key]
            )
            inst.notes = all_notes[key]
            inst.control_changes = all_ccs[key]
            inst.pitch_bends = all_pbends[key]
            final_instruments.append(inst)

    for key in drum_keys:
        inst = pretty_midi.Instrument(
            program=key[0], is_drum=key[1], name=instrument_names[key]
        )
        inst.notes = all_notes[key]
        inst.control_changes = all_ccs[key]
        inst.pitch_bends = all_pbends[key]
        final_instruments.append(inst)

    master_pm.instruments = final_instruments
    for inst in master_pm.instruments:
        inst.notes.sort(key=lambda note: note.start)
        inst.control_changes.sort(key=lambda x: x.time)
        inst.pitch_bends.sort(key=lambda x: x.time)
    master_pm.write(str(output_file))


def resolve_stem_paths(*, song_path, stem_dir, stem_names):
    """Reconstruct standard stem output paths, returning only those that exist."""
    song_file = Path(song_path)
    song_id = song_file.stem
    stem_root = Path(stem_dir) / song_id
    resolved = {}

    for stem_name in stem_names:
        expected_path = stem_root / f"{song_id}_{stem_name}.wav"
        if expected_path.exists():
            resolved[stem_name] = expected_path

    if resolved:
        return resolved

    if not stem_root.exists():
        return resolved

    for wav_path in sorted(stem_root.glob("*.wav")):
        stem_key = wav_path.stem
        for stem_name in stem_names:
            if stem_key.endswith(f"_{stem_name}"):
                resolved.setdefault(stem_name, wav_path)
                break
    return resolved


def prepare_audio_for_stem_separation(audio_path, *, temp_dir):
    """Coerce the input audio to a 2-channel WAV for the separator."""
    audio_file = Path(audio_path)
    waveform, sample_rate = librosa.load(str(audio_file), sr=None, mono=False)

    if waveform.ndim == 1:
        waveform = waveform[None, :]
    elif waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
        waveform = waveform.T

    source_channels = int(waveform.shape[0])
    if source_channels <= 0:
        raise ValueError(f"Audio file has no channels: {audio_file}")
    if source_channels == 2:
        return audio_file

    if source_channels == 1:
        waveform = np.repeat(waveform, 2, axis=0)
        channel_mode = "pseudo-stereo"
    else:
        waveform = waveform[:2]
        channel_mode = "first-two-channels"

    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = temp_dir / f"{audio_file.stem}.wav"
    sf.write(str(prepared_path), waveform.T, samplerate=int(sample_rate))
    print(
        f"Prepared {channel_mode} input for stem separation: "
        f"source_channels={source_channels} -> {prepared_path.name}"
    )
    return prepared_path


def get_stem_pipeline_models(checkpoint_path=None, device_preference=None, model_type="default"):
    """Load the AMT model + separation model, reusing them across the session."""
    device = torch.device(device_preference or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = infer._ensure_checkpoint(
        None if checkpoint_path in (None, "", "DEFAULT") else Path(checkpoint_path),
        model_type=model_type,
    )
    amt_cache_key = ("amt", str(checkpoint.resolve()), device.type)
    sep_cache_key = ("sep", device.type)

    if sep_cache_key not in STEM_PIPELINE_CACHE:
        print(f"Loading separation model on {device} ...")
        sep_config = SeparationConfig(skip_existing=True)
        sep_model = load_mss_model(sep_config, device=device)
        sep_dtype = (
            torch.float16
            if sep_config.use_half_precision and device.type == "cuda"
            else torch.float32
        )
        STEM_PIPELINE_CACHE[sep_cache_key] = (sep_config, sep_model, sep_dtype)
    else:
        sep_config, sep_model, sep_dtype = STEM_PIPELINE_CACHE[sep_cache_key]

    if amt_cache_key not in STEM_PIPELINE_CACHE:
        print(f"Loading AMT model ({model_type}) on {device} ...")
        amt_model, amt_config, amt_settings = infer._load_model_and_settings(
            checkpoint,
            device=device,
            window_ms_override=None,
            stride_ms_override=None,
            track_batch_size_override=None,
        )
        STEM_PIPELINE_CACHE[amt_cache_key] = (amt_model, amt_config, amt_settings)
    else:
        print(f"Reusing cached AMT model ({model_type}) on {device} ...")
        amt_model, amt_config, amt_settings = STEM_PIPELINE_CACHE[amt_cache_key]

    return {
        "device": device,
        "checkpoint": checkpoint,
        "amt_model": amt_model,
        "amt_config": amt_config,
        "amt_settings": amt_settings,
        "sep_config": sep_config,
        "sep_model": sep_model,
        "sep_dtype": sep_dtype,
    }


def resolve_stem_model_type(stem_name):
    """Choose the AMT model type for a separated stem (matches the Colab)."""
    stem_name = stem_name.lower()
    if "drum" in stem_name:
        return "drums"
    if "bass" in stem_name:
        return "bass"
    if "vocal" in stem_name:
        return "vocal_harmony"
    if "guitar" in stem_name:
        return "guitar"
    if "other" in stem_name:
        return "other"
    return "default"


def run_stem_separated_transcription(
    audio_path,
    *,
    output_midi,
    checkpoint_path=None,
    output_root="stem_outputs",
    device_preference=None,
    window_batch_size=4,
    max_midi_melodic_instruments=15,
    transcribe_drum_stems=True,
    cleanup_separated_stems=False,
    merge_onset_ms=20.0,
    force=False,
):
    """Separate -> transcribe each stem -> merge, writing the result to output_midi."""
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    bundle = get_stem_pipeline_models(
        checkpoint_path=checkpoint_path, device_preference=device_preference, model_type="default"
    )
    device = bundle["device"]
    sep_config = bundle["sep_config"]
    sep_model = bundle["sep_model"]
    sep_dtype = bundle["sep_dtype"]

    amt_bundles = {"default": bundle}

    def get_amt_bundle(model_type):
        if model_type not in amt_bundles:
            amt_bundles[model_type] = get_stem_pipeline_models(
                checkpoint_path=checkpoint_path,
                device_preference=device_preference,
                model_type=model_type,
            )
        return amt_bundles[model_type]

    run_root = Path(output_root)
    stem_dir = run_root
    stem_midi_dir = run_root / "midi"
    prepared_dir = run_root / "prepared"
    for directory in (stem_dir, stem_midi_dir):
        directory.mkdir(parents=True, exist_ok=True)

    separation_input = prepare_audio_for_stem_separation(audio_file, temp_dir=prepared_dir)
    print(f"Separating stems for: {audio_file.name}")
    stems = _separate_one_file(
        separation_input, stem_dir, sep_config, sep_model, device, sep_dtype
    )

    if not stems:
        stems = resolve_stem_paths(
            song_path=audio_file, stem_dir=stem_dir, stem_names=sep_config.stem_names
        )
        if stems:
            print(f"Reusing existing stems: {sorted(stems)}")
        else:
            raise RuntimeError(f"No stems found for {audio_file.stem}")

    song_midi_paths = []
    for stem_name, stem_path in sorted(stems.items()):
        if not transcribe_drum_stems and "drum" in stem_name.lower():
            print(f"Skipping drum stem: {stem_name}")
            continue

        out_stem_midi = stem_midi_dir / f"{audio_file.stem}_{stem_name}.mid"
        if out_stem_midi.exists() and not force:
            print(f"Reusing existing stem MIDI: {stem_name}")
            song_midi_paths.append(out_stem_midi)
            continue

        model_type = resolve_stem_model_type(stem_name)
        current_bundle = get_amt_bundle(model_type)

        print(f"Transcribing stem: {stem_name} (model={model_type})")
        waveform, _, _ = infer._load_audio(
            Path(stem_path), target_sample_rate=current_bundle["amt_config"].sample_rate
        )
        notes, _, _ = infer.run_inference(
            model=current_bundle["amt_model"],
            waveform=waveform.to(device),
            model_config=current_bundle["amt_config"],
            settings=current_bundle["amt_settings"],
            device=device,
            amp_enabled=False,
            amp_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            velocity=100,
            merge_gap_ms=None,
            merge_onset_ms=merge_onset_ms,
            silence_gate_rms_dbfs=-72,
            window_batch_size=window_batch_size,
            max_midi_melodic_instruments=max_midi_melodic_instruments,
            disable_tqdm=True,
            max_note_seconds=15.0,
        )

        midi = infer._build_midi(
            notes,
            sample_rate=current_bundle["amt_config"].sample_rate,
            instrument_volumes=dict(infer.DEFAULT_INSTRUMENT_VOLUMES),
        )
        midi.write(str(out_stem_midi))
        song_midi_paths.append(out_stem_midi)

    if not song_midi_paths:
        raise RuntimeError("No stem MIDI files were generated")

    output_midi = Path(output_midi)
    output_midi.parent.mkdir(parents=True, exist_ok=True)
    merge_midis_logic(song_midi_paths, output_midi, max_melodic=max_midi_melodic_instruments)

    if cleanup_separated_stems:
        shutil.rmtree(stem_dir, ignore_errors=True)

    print(f"stem_midis_dir={stem_midi_dir}")
    print(f"merged_midi={output_midi}")
    print(f"Merged MIDI written to {output_midi}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stem-separated AMT transcription.")
    parser.add_argument("--amt-repo", required=True, help="Path to the AMT repo checkout.")
    parser.add_argument("--audio", required=True, help="Input audio file.")
    parser.add_argument("--output-midi", required=True, help="Path for the merged MIDI.")
    parser.add_argument("--output-root", default="stem_outputs", help="Intermediate output dir.")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto).")
    parser.add_argument("--checkpoint", default=None, help="Optional AMT checkpoint override.")
    parser.add_argument("--window-batch-size", type=int, default=4)
    parser.add_argument("--max-midi-melodic-instruments", type=int, default=15)
    parser.add_argument("--merge-onset-ms", type=float, default=20.0)
    parser.add_argument(
        "--no-transcribe-drums", action="store_true", help="Skip the drum stem."
    )
    parser.add_argument(
        "--cleanup-stems", action="store_true", help="Delete separated stem WAVs when done."
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-transcribe stems even if their MIDIs exist."
    )
    args = parser.parse_args()

    run_stem_separated_transcription(
        args.audio,
        output_midi=args.output_midi,
        checkpoint_path=args.checkpoint,
        output_root=args.output_root,
        device_preference=args.device,
        window_batch_size=args.window_batch_size,
        max_midi_melodic_instruments=args.max_midi_melodic_instruments,
        transcribe_drum_stems=not args.no_transcribe_drums,
        cleanup_separated_stems=args.cleanup_stems,
        merge_onset_ms=args.merge_onset_ms,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
