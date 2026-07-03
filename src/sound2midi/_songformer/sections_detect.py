"""Detect song structure (intro / verse / chorus / ...) with SongFormer.

Runs INSIDE the SongFormer venv (torch, muq, musicfm and the repo's own modules
live there), invoked by ``sound2midi.sections.detect_sections``. Not part of this
package's lint/type-check surface.

This is a faithful headless port of the upstream single-file inference path
(``app.py`` / ``infer/infer.py`` in https://github.com/ASLP-lab/SongFormer):
MuQ + MusicFM embeddings over 420 s windows (plus 30 s sub-windows), fused and
fed to the SongFormer MSA head, post-processed to labeled segments using the
SongForm-HX-8Class label set (intro, verse, pre-chorus, chorus, bridge, inst,
outro, silence). Differences from upstream:

  * no gradio / matplotlib / multiprocessing;
  * ``--device cpu`` supported, and CUDA falls back to CPU when unusable
    (e.g. a torch build without kernels for this GPU);
  * the trailing sub-1024-sample window breaks out of the loop instead of
    spinning forever (upstream ``continue`` bug);
  * segments are written as one artifact JSON with floats, not strings.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# --- upstream inference constants (app.py / infer.py) ---
MUSICFM_HOME_PATH = os.path.join("ckpts", "MusicFM")
AFTER_DOWNSAMPLING_FRAME_RATES = 8.333
DATASET_LABEL = "SongForm-HX-8Class"
DATASET_IDS = [5]
TIME_DUR = 420
INPUT_SAMPLING_RATE = 24000
WIN_SIZE = 420
HOP_SIZE = 420
NUM_CLASSES = 128
MODEL_NAME = "SongFormer"
CHECKPOINT = "SongFormer.safetensors"
CONFIG = "SongFormer.yaml"


def pick_device(requested: str) -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        print("CUDA is not available; running on CPU (slow).", file=sys.stderr)
        return "cpu"
    try:  # smoke-test the kernels: a torch built for older GPUs imports fine but can't run
        (torch.ones(2, 2, device="cuda") @ torch.ones(2, 2, device="cuda")).sum().item()
    except RuntimeError as exc:
        print(f"CUDA is unusable on this GPU ({exc}); running on CPU (slow).", file=sys.stderr)
        return "cpu"
    return "cuda"


def load_checkpoint(checkpoint_path: str, device: str = "cpu"):
    import torch

    if checkpoint_path.endswith(".pt"):
        return torch.load(checkpoint_path, map_location=device)
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return {"model_ema": load_file(checkpoint_path, device=device)}
    raise ValueError("Unsupported checkpoint format. Use .pt or .safetensors")


def load_models(device: str):
    """Load MuQ, MusicFM and the SongFormer MSA model (upstream initialize_models)."""
    from ema_pytorch import EMA
    from muq import MuQ
    from musicfm.model.musicfm_25hz import MusicFM25Hz
    from omegaconf import OmegaConf

    print("Loading MuQ (OpenMuQ/MuQ-large-msd-iter) ...", file=sys.stderr)
    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    muq = muq.to(device).eval()

    print("Loading MusicFM ...", file=sys.stderr)
    musicfm = MusicFM25Hz(
        is_flash=False,
        stat_path=os.path.join(MUSICFM_HOME_PATH, "msd_stats.json"),
        model_path=os.path.join(MUSICFM_HOME_PATH, "pretrained_msd.pt"),
    )
    musicfm = musicfm.to(device).eval()

    print("Loading SongFormer ...", file=sys.stderr)
    module = importlib.import_module("models." + MODEL_NAME)
    hp = OmegaConf.load(os.path.join("configs", CONFIG))
    model = getattr(module, "Model")(hp)

    ckpt = load_checkpoint(os.path.join("ckpts", CHECKPOINT))
    if ckpt.get("model_ema") is not None:
        model_ema = EMA(model, include_online_model=False)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(model_ema.ema_model.state_dict())
    else:
        model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    return muq, musicfm, model, hp


def process_audio(audio_path: str, device: str, muq, musicfm, model, hp):
    """Windowed embedding extraction + MSA inference (upstream process_audio)."""
    import librosa
    import torch
    from dataset.label2id import DATASET_ID_ALLOWED_LABEL_IDS, DATASET_LABEL_TO_DATASET_ID
    from postprocessing.functional import postprocess_functional_structure

    wav, _sr = librosa.load(audio_path, sr=INPUT_SAMPLING_RATE)
    audio = torch.tensor(wav).to(device)

    total_len = ((audio.shape[0] // INPUT_SAMPLING_RATE) // TIME_DUR) * TIME_DUR + TIME_DUR
    total_frames = math.ceil(total_len * AFTER_DOWNSAMPLING_FRAME_RATES)

    logits = {
        "function_logits": np.zeros([total_frames, NUM_CLASSES]),
        "boundary_logits": np.zeros([total_frames]),
    }
    logits_num = {
        "function_logits": np.zeros([total_frames, NUM_CLASSES]),
        "boundary_logits": np.zeros([total_frames]),
    }

    dataset_id2label_mask = {}
    for key, allowed_ids in DATASET_ID_ALLOWED_LABEL_IDS.items():
        dataset_id2label_mask[key] = np.ones(NUM_CLASSES, dtype=bool)
        dataset_id2label_mask[key][allowed_ids] = False

    lens = 0
    i = 0
    with torch.no_grad():
        while True:
            start_idx = i * INPUT_SAMPLING_RATE
            end_idx = min((i + WIN_SIZE) * INPUT_SAMPLING_RATE, audio.shape[-1])
            if start_idx >= audio.shape[-1]:
                break
            if end_idx - start_idx <= 1024:
                break  # upstream `continue`s here, which would loop forever
            audio_seg = audio[start_idx:end_idx]

            muq_output = muq(audio_seg.unsqueeze(0), output_hidden_states=True)
            muq_embd_420s = muq_output["hidden_states"][10]
            del muq_output
            torch.cuda.empty_cache()

            _, musicfm_hidden_states = musicfm.get_predictions(audio_seg.unsqueeze(0))
            musicfm_embd_420s = musicfm_hidden_states[10]
            del musicfm_hidden_states
            torch.cuda.empty_cache()

            wraped_muq_embd_30s = []
            wraped_musicfm_embd_30s = []
            for idx_30s in range(i, i + HOP_SIZE, 30):
                start_idx_30s = idx_30s * INPUT_SAMPLING_RATE
                end_idx_30s = min(
                    (idx_30s + 30) * INPUT_SAMPLING_RATE,
                    audio.shape[-1],
                    (i + HOP_SIZE) * INPUT_SAMPLING_RATE,
                )
                if start_idx_30s >= audio.shape[-1]:
                    break
                if end_idx_30s - start_idx_30s <= 1024:
                    continue
                wraped_muq_embd_30s.append(
                    muq(
                        audio[start_idx_30s:end_idx_30s].unsqueeze(0),
                        output_hidden_states=True,
                    )["hidden_states"][10]
                )
                torch.cuda.empty_cache()
                wraped_musicfm_embd_30s.append(
                    musicfm.get_predictions(audio[start_idx_30s:end_idx_30s].unsqueeze(0))[1][10]
                )
                torch.cuda.empty_cache()

            if wraped_muq_embd_30s:
                wraped_muq_embd_30s = torch.concatenate(wraped_muq_embd_30s, dim=1)
                wraped_musicfm_embd_30s = torch.concatenate(wraped_musicfm_embd_30s, dim=1)
                all_embds = [
                    wraped_musicfm_embd_30s,
                    wraped_muq_embd_30s,
                    musicfm_embd_420s,
                    muq_embd_420s,
                ]

                embd_lens = [x.shape[1] for x in all_embds]
                min_embd_len = min(embd_lens)
                if max(embd_lens) - min_embd_len > 4:
                    raise ValueError(
                        f"Embedding shapes differ too much: {max(embd_lens)} vs {min_embd_len}"
                    )
                all_embds = [x[:, :min_embd_len, :] for x in all_embds]
                embd = torch.concatenate(all_embds, axis=-1)

                dataset_ids = torch.Tensor(DATASET_IDS).to(device, dtype=torch.long)
                _msa_info, chunk_logits = model.infer(
                    input_embeddings=embd,
                    dataset_ids=dataset_ids,
                    label_id_masks=torch.Tensor(
                        dataset_id2label_mask[DATASET_LABEL_TO_DATASET_ID[DATASET_LABEL]]
                    )
                    .to(device, dtype=bool)
                    .unsqueeze(0)
                    .unsqueeze(0),
                    with_logits=True,
                )

                start_frame = int(i * AFTER_DOWNSAMPLING_FRAME_RATES)
                end_frame = start_frame + min(
                    math.ceil(HOP_SIZE * AFTER_DOWNSAMPLING_FRAME_RATES),
                    chunk_logits["boundary_logits"][0].shape[0],
                )
                logits["function_logits"][start_frame:end_frame, :] += (
                    chunk_logits["function_logits"][0].detach().cpu().numpy()
                )
                logits["boundary_logits"][start_frame:end_frame] = (
                    chunk_logits["boundary_logits"][0].detach().cpu().numpy()
                )
                logits_num["function_logits"][start_frame:end_frame, :] += 1
                logits_num["boundary_logits"][start_frame:end_frame] += 1
                lens += end_frame - start_frame

            i += HOP_SIZE

    logits["function_logits"] /= np.maximum(logits_num["function_logits"], 1)
    logits["boundary_logits"] /= np.maximum(logits_num["boundary_logits"], 1)
    logits["function_logits"] = torch.from_numpy(logits["function_logits"][:lens]).unsqueeze(0)
    logits["boundary_logits"] = torch.from_numpy(logits["boundary_logits"][:lens]).unsqueeze(0)

    return postprocess_functional_structure(logits, hp)


def rule_post_processing(msa_list):
    """Upstream rule-based cleanup: merge sliver segments at the song edges."""
    if len(msa_list) <= 2:
        return msa_list

    result = msa_list.copy()

    while len(result) > 2:
        first_duration = result[1][0] - result[0][0]
        if first_duration < 1.0 and len(result) > 2:
            result[0] = (result[0][0], result[1][1])
            result = [result[0]] + result[2:]
        else:
            break

    while len(result) > 2:
        last_label_duration = result[-1][0] - result[-2][0]
        if last_label_duration < 1.0:
            result = result[:-2] + [result[-1]]
        else:
            break

    while len(result) > 2:
        if result[0][1] == result[1][1] and result[1][0] <= 10.0:
            result = [(result[0][0], result[0][1])] + result[2:]
        else:
            break

    while len(result) > 2:
        last_duration = result[-1][0] - result[-2][0]
        if result[-2][1] == result[-3][1] and last_duration <= 10.0:
            result = result[:-2] + [result[-1]]
        else:
            break

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect song structure with SongFormer.")
    parser.add_argument("--repo", required=True, help="Path to the SongFormer checkout.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    audio = Path(args.audio).resolve()
    if not audio.is_file():
        print(f"Audio file not found: {audio}", file=sys.stderr)
        return 2

    output_json = Path(args.output_json).resolve() if args.output_json else None

    # The repo's modules and its ckpts/configs paths are all relative to src/SongFormer.
    repo = Path(args.repo).resolve()
    pkg_root = repo / "src" / "SongFormer"
    os.chdir(pkg_root)
    sys.path.insert(0, str(repo / "src" / "third_party"))  # musicfm (git submodule)
    sys.path.insert(0, str(pkg_root))

    # Upstream monkeypatch: msaf still uses the long-removed scipy.inf alias.
    import scipy

    scipy.inf = np.inf

    device = pick_device(args.device)
    muq, musicfm, model, hp = load_models(device)

    msa_output = process_audio(str(audio), device, muq, musicfm, model, hp)
    if not msa_output or msa_output[-1][-1] != "end":
        raise RuntimeError(f"Unexpected SongFormer output (no end marker): {msa_output!r}")
    msa_output = rule_post_processing(msa_output)

    segments = [
        {
            "label": msa_output[idx][1],
            "start": round(float(msa_output[idx][0]), 2),
            "end": round(float(msa_output[idx + 1][0]), 2),
        }
        for idx in range(len(msa_output) - 1)
    ]
    structure = " ".join(seg["label"] for seg in segments)

    result = {
        "segments": segments,
        "structure": structure,
        "duration": segments[-1]["end"] if segments else 0.0,
        "label_set": DATASET_LABEL,
        "model": "songformer",
        "device": device,
        "audio": str(audio),
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2) + "\n")

    summary = {
        "n_segments": len(segments),
        "structure": structure,
        "duration": result["duration"],
        "device": device,
    }
    print(json.dumps(summary))  # parseable result line for the parent process
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
