# Spec — MPE Expression Engine for `vibetuned/sound2midi`

Target repo: https://github.com/vibetuned/sound2midi (main). Follow the house style: `uv` packaging, `ruff` lint/format, `ty` type-check — all three must pass. Heavy/optional deps go behind an extra, like `--extra player` does.

## 1. Context and goal

sound2midi transcribes audio to MIDI and produces per-song JSON artifacts in `output/<song>/artifacts/`: `key.json` (skey), `meter.json` (Beat This! — tempo, time signature, **full beat/downbeat grid**), `sections.json` (SongFormer segments, opt-in), `chords.json` (lv-chordia, Harte labels, opt-in). Score-derived or transcribed MIDI plays flat on MPE synths: constant/no velocity, no per-note expression.

Build `sound2midi/expression/`: an engine that takes a MIDI file + the song's artifacts and generates:

1. **Velocity** (synthesized from musical context, or from stem loudness when stems exist)
2. **Channel Pressure** (per-voice aftertouch)
3. **CC74** (timbre/brightness)
4. **Pitch Bend** (vibrato)

Two outputs: (a) an **MPE `.mid` file** written next to the source (`<name>.mpe.mid`), and (b) a **realtime virtual MIDI port** streaming MPE to a DAW. Note: the existing player uses FluidSynth, which is not MPE-capable — do **not** touch the player engine; the realtime path is a separate CLI. Player integration is a non-goal (v2 idea: an "MPE out" engine option).

> **Disambiguation — MPE ≠ MEI.** This spec is about **MPE** (*MIDI Polyphonic Expression*: per-note channels carrying pressure/CC74/bend). The repo's existing `sound2midi-mei` tool is unrelated: **MEI** is the *Music Encoding Initiative* notation format (MusicXML → MEI via Verovio, consumed by midi-stroke). Do not modify, import from, or "unify" anything under the MEI/notation path. The two acronyms differ by one letter; keep module and CLI names visibly distinct (`expression/`, `sound2midi-mpe*`).

New extra + entry points in `pyproject.toml`, following the existing pattern:

```toml
[project.optional-dependencies]
mpe = ["python-rtmidi>=1.5", "numpy", "soundfile", "pyloudnorm"]
# mido is already a core dep; prefer soundfile+numpy over librosa (no numba pull)

[project.scripts]
sound2midi-mpe      = "sound2midi.expression.cli:render_main"
sound2midi-mpe-play = "sound2midi.expression.cli:play_main"
```

CLI shape (mirror existing tools):

```
uv run sound2midi-mpe output/<id>/<id>.stems.mid            # -> <id>.stems.mpe.mid
uv run sound2midi-mpe song.mid --profile strings --bend-range 48 --seed 7
uv run sound2midi-mpe-play output/<id>/<id>.stems.mpe.mid   # stream to DAW
uv run sound2midi-mpe-play output/<id>/<id>.stems.mid --live # render on the fly
```

Artifacts are discovered automatically from the song-folder layout (`artifacts/<song>.*.json` relative to the MIDI); every artifact is optional with graceful fallback, and `--artifacts-dir` overrides discovery.

---

## 2. Module A — `context.py`: per-note features from the artifacts

**Input:** `mido.MidiFile` (Type 0/1) + optional artifact dicts.

Group notes per track (a track = a voice/instrument, matching the player's lane model). For each note emit a feature record:

| Feature | Range | Source |
|---|---|---|
| `duration_s` (T_i) | s | note events, using the file's tempo map |
| `interval_st` (Δp_i) | signed st | previous note **in the same track**; for polyphonic tracks track the top voice (highest concurrent note), like the player's `1v` logic; first note of a phrase → 0 |
| `metric_weight` (M_i) | [0,1] | **`meter.json` beat grid**: map note onset (seconds) to beat position with the same piecewise-linear seconds→beats mapping the exporter uses (reuse/extract that code from `player/export.py` into a shared helper rather than duplicating). Downbeat 1.0; mid-bar strong beat 0.75 (beat 3 in 4/4, beat 4 in 6/8); other beats 0.5; 8th offbeat 0.25; else 0.1. Fallback without `meter.json`: MIDI PPQ grid, assume 4/4. |
| `tension` (H_i) | [0,1] | **`chords.json`**: find the Harte chord active at onset (`C:maj`, `A:min7`, `E:maj/3`, `N`). Parse with the existing Harte parsing (`player/chordlabel`) — extract it to a non-player module (e.g. `sound2midi/harte.py`) so the `mpe` extra never imports PySide6. Map pitch class vs. chord: root/5th = 0.0, 3rd = 0.2, 6th/maj7 = 0.4, 7th/9th/sus = 0.7, chromatic non-chord or tritone-vs-root = 1.0; during `N` spans, fall back to `key.json` diatonic distance (tonic/5th 0.0, other diatonic 0.3, chromatic 0.8). No artifacts at all → H_i = 0.3 constant. |
| `phrase_pos` (φ_i) | [0,1] | phrase segmentation below |
| `section` | enum + [0,1] | **`sections.json`** segment label at onset (verse/chorus/bridge/intro/outro/inst/silence) + intensity scalar per label (configurable; defaults: intro 0.7, verse 0.85, pre-chorus 0.95, chorus 1.0, bridge 0.9, inst 0.95, outro 0.7). Fallback: 0.9 everywhere. |

**Phrase segmentation (required):** per track, split on: rest ≥ 1 beat (beat length from meter.json tempo), OR note ≥ 2× local median duration followed by any gap, OR a section boundary. Phrase arc `Φ(φ) = 0.6 + 0.4·sin(π·φ^0.9)`, multiplied into velocity, swell amplitude, and vibrato depth. This is the highest-impact musicality feature — do not skip.

**Drums:** skip expression entirely for channel 10 / drum-named tracks (the stems pipeline produces a real drum track); velocity synthesis only, no pressure/bend/CC74.

---

## 3. Module B — `velocity.py`: velocity synthesis

Transcribed MIDI may carry flat or fixed velocity (`--infer-arg=--velocity`); score-derived MIDI has none. Synthesize:

```
vel_i = clamp(20, 127,
   ( V_base·section_intensity          # V_base default 72
   + 30·M_i
   + 18·min(1, |Δp_i|/12)·(1.25 if Δp_i>0 else 1.0)
   + 10·H_i
   ) · Φ(φ_i) · (1 + N(0, 0.04)) )
```

Rules: notes < 120 ms get −8; phrase pitch-peak gets +6; repeated identical pitches alternate ±4; seedable RNG. `--keep-velocity` skips synthesis when the source velocities are meaningful. When stem loudness is available (Module F): `vel = w·vel_model + (1−w)·vel_loudness`, `w` default 0.3 (`--vel-blend`).

---

## 4. Module C — `curves.py`: expression curve generation

Per note, sample three streams at 60 Hz internally (emission is delta-thresholded in Module D). Base model per `midisoul.md`, with these **mandatory corrections/additions**:

### 4.1 Pressure P(t)
```
P(t) = 18 + A_swell·sin^1.2(π·u^0.85) + A_att·exp(−t/0.03)
A_swell = Φ(φ_i)·section_intensity·45·(0.4·M_i + 0.35·min(1,|Δp_i|/12) + 0.25·H_i)·jitter
A_att   = 25·(vel_i/96)
```
Plus slow drift: low-pass-filtered gaussian noise (~1.5 Hz cutoff, ±4) so held notes never sit on a perfect curve.

### 4.2 CC74 C(t)
```
C(t) = clamp(0,127, C_floor + 0.55·P(t) + 25·H_i + 2·max(0,Δp_i)·exp(−t/0.08))
```

### 4.3 Vibrato V(t) — cents
- Gate: none if `T_i < 0.35 s`; onset `t_onset = min(0.3, 0.35·T_i)` (±30 ms jitter).
- Depth: sigmoid ramp to `A_max = Φ(φ_i)·35·(0.6+0.4·H_i)` cents (±10% jitter), with depth wobble `×(1 + 0.08·sin(2π·0.7·t + rand_phase))`.
- Rate: `f_v = 4.8 + 1.4·u + r_note` Hz, `r_note ~ N(0,0.3)` drawn once per note.
- **Accumulate phase** — never `sin(2π·f_v(t)·t)`:
  ```python
  phase += 2*math.pi*f_v*dt
  cents = A_v * math.sin(phase)
  ```

### 4.4 Humanization
All per-note jitters drawn from a seedable RNG (`--seed`). Two identical consecutive notes must never yield identical curves; with a fixed seed, output is byte-reproducible (both unit-tested).

### 4.5 Instrument profiles — `profiles/*.yaml`
Profiles = stream enables + weight overrides. **Auto-select from stem track names** when rendering a `.stems.mid` (track names come from the stem merge): `vocals→sung`, `guitar→pluck`, `bass→pluck`, `piano→keys`, `other→strings`, `drums→none`. `--profile` forces one globally; `--track-profile NAME=PROFILE` overrides per track.

| profile | pressure | vibrato | CC74 |
|---|---|---|---|
| `strings` | full | full | full |
| `sung` | full, slower attack | full, deeper (A_max ×1.3), later onset | full |
| `winds` | full, β=0.7 | full | full |
| `pluck`/`keys` | attack transient + decay only (no swell) | off | tracks decay |
| `pads` | slow swell, no attack | shallow slow | full |
| `none` | — | — | — |

---

## 5. Module D — `mpe_encoder.py` (shared by file + realtime)

1. **MPE handshake:** master ch 1 (index 0): RPN 6 → 15 member channels (`CC101=0, CC100=6, CC6=15`); then RPN 0 pitch-bend sensitivity = `--bend-range` (default **48** st) on the member channels.
2. **Channel allocation:** round-robin over member channels 2–16 (indices 1–15), per MPE zone. All busy → steal oldest note (emit its note_off first).
3. **Per note** on its member channel: `note_on(vel)` → interleaved `aftertouch` / `control_change 74` / `pitchwheel` → `note_off`, then resets: `pitchwheel=0` (mido center is **0**, not 8192), pressure 0, CC74 → floor.
4. **Bend conversion:** `bend = round(cents/(100·bend_range)·8192)` clamped to `[−8192, 8191]`.
5. **Delta thresholding:** emit only on change (pressure/CC74 ≥ 1 step, bend ≥ 16 units); target ≥60% event reduction vs. naive 60 Hz.
6. **File export:** Type 1, PPQ 480 (retick from source), tempo map preserved so the MPE file stays aligned with `meter.json` and the source audio. One MPE zone; if the source has > 15 simultaneously sounding voices, log a warning about channel stealing.

## 6. Module E — `realtime.py` + `cli.py`: DAW test source

- `sound2midi-mpe-play`: open a **virtual output port** `sound2midi MPE` via `python-rtmidi` (macOS/Linux native; on Windows virtual ports aren't supported — detect and print loopMIDI instructions).
- Send handshake on open; stream with a monotonic-clock scheduler (sleep-to-next-event, jitter < 3 ms).
- Flags: `--profile`, `--track-profile`, `--bend-range`, `--seed`, `--rubato`, `--legato`, `--solo-track N`, `--loop`, `--start SECONDS`, `--live` (render from a plain MIDI on the fly), and `--dry` (velocity only, expression muted) for A/B.
- Ctrl-C panic: all-notes-off + bend/pressure reset on every member channel.

## 7. Module F — `loudness.py`: dynamics from stems

The stems WAVs already exist at `output/<id>/stems/` and share the source audio's timeline with the per-stem MIDIs — no extra alignment needed.

1. Per stem WAV (read with `soundfile`): short-term RMS dB, 50 ms window / 10 ms hop (numpy; no librosa). Optional LUFS-S via `pyloudnorm` (`--loudness lufs`).
2. Normalize per stem to [0,1] by rolling percentiles (5th→0, 98th→1).
3. **Velocity:** max of the curve over the note's first 80 ms → `vel = 20 + 100·x^0.8`, blended per Module B.
4. **Pressure (opt-in `--loudness-pressure`):** resample the loudness curve across the note and blend 50/50 with synthetic P(t), so real swells survive into MPE.
5. Match stem WAV ↔ MIDI track by the stem name embedded in filenames/track names by the existing merge step.

## 8. Module G — `timing.py`: rubato / expressive timing

Flag `--rubato STRENGTH` (0–1, default 0.5; `--rubato 0` disables). Applied **before** curve generation (Module C) so vibrato onsets, swells and phrase arcs track the displaced timings — never after.

Two layers:

1. **Global agogic warp — all tracks share one monotonic time-warp `w(t)`**, so the ensemble stays together:
   - Phrase-level: slight acceleration into the phrase peak (up to +2%·strength local tempo), ritenuto on the last 1–2 beats of a phrase (up to −4%·strength).
   - Bar-level agogic accent: downbeats lengthened ~1%·strength.
   - Section boundaries: broadening of up to 60 ms·strength on the last beat before a chorus/section change (from `sections.json`).
   - **Drift cap:** `|w(t) − t| ≤ 80 ms·strength` at all times, so the render never wanders audibly off the `meter.json` grid or the source stems.
2. **Per-voice asynchrony (after the warp):**
   - Melody lead: the track carrying the phrase's top voice plays 10–25 ms early on strong beats relative to accompaniment.
   - Onset jitter: gaussian σ = 6 ms·strength per note.
   - Releases: inner notes shortened −3%; phrase-final notes lengthened +5%.

Implementation rules: displace **event times**, do **not** rewrite the tempo map (keeps DAW import and `meter.json`-relative alignment predictable). All randomness from the seeded RNG. Heuristic default: when the source MIDI is a raw transcription (micro-timing already human), auto-suggest `--rubato 0` in the log — rubato synthesis is mainly for score-quantized input; quantized input is detected by onset-to-grid deviation < 10 ms median.

## 9. Module H — legato & portamento (in `curves.py` + `mpe_encoder.py`)

Flag `--legato MODE` (`auto`|`glide`|`overlap`|`off`; default `auto` = per profile: on for `sung`/`strings`/`winds`, off for `pluck`/`keys`/`none`).

**Detection:** within a phrase, consecutive notes qualify when inter-onset gap ≤ 60 ms (measured after rubato) AND |Δp| ≤ profile threshold (default 4 st; `sung` 7 st).

**`glide` mode (the MPE-native one, default for `sung`):** a legato group is rendered as **one sustained MPE note** on one member channel: note_on at the first pitch, then at each inner transition the pitch bend **travels** to the next pitch over `t_glide = clamp(30, 80, 12·|Δp|)` ms (S-curve, not linear) and stays there; vibrato continues relative to the current bend offset; each inner transition adds a pressure bump (+10–15, decaying 80 ms) and a CC74 transient so articulation survives. note_off only at the group's end, then full resets.
   - **Bend budget:** cumulative offset from the group's root must stay within ±(bend_range − 2) st; exceeding it breaks the group (retrigger a new note). This is why `--bend-range 48` is the default — at ±2 the feature is effectively unusable, and the encoder must log a warning if `--legato glide` is combined with a small bend range.
   - Document clearly: a `.mpe.mid` with glides is a *performance render* — inner notes of a chain no longer exist as note events, so it is not a notation/re-edit source.

**`overlap` mode (default for `strings`/`winds`):** keep per-note channels but extend each note to overlap its successor by 20–40 ms, and suppress the successor's attack transient (A_att ×0.3) and reduce its velocity toward the phrase mean — bowed/blown connection without bend chains. Safe with any bend range.

Velocity interaction: inner legato notes get −10 velocity (soft attacks); Module B applies this after its own rules.

## 10. Layout, tooling, tests

```
src/sound2midi/expression/
  __init__.py  context.py  velocity.py  curves.py
  mpe_encoder.py  realtime.py  loudness.py  cli.py
  profiles/*.yaml
src/sound2midi/harte.py        # Harte parsing extracted from player/chordlabel (player re-imports it)
tests/expression/              # pytest; add pytest to dev deps if absent
```

Constraints: `uv run ruff check`, `uv run ruff format --check`, `uv run ty check` all clean. The `mpe` extra must not import PySide6/FluidSynth; core install stays lightweight (mido only), everything else behind `--extra mpe`. No changes to the AMT/SongFormer venv machinery.

## 11. Acceptance criteria

1. `uv run sound2midi-mpe output/<id>/<id>.stems.mid` (with `meter/chords/sections` artifacts present) produces `<id>.stems.mpe.mid` that plays in an MPE synth (Vital, Surge XT, Bitwig/Ableton MPE) with per-note bend/pressure, no stuck bends, no pressure bleed across channel reuse.
2. Same command with **zero artifacts** still works (fallback features), only logs which artifacts were missing.
3. Bend math: 100-cent peak at range 48 → bend ±171 (±1); resets present at every note_off; all bends within [−8192, 8191].
4. No chirp: on a 4 s note the measured instantaneous vibrato rate stays in [4.5, 6.5] Hz (verifies phase accumulation).
5. Humanization: identical consecutive notes → non-identical curves; fixed `--seed` → byte-identical file across runs.
6. Delta thresholding: ≥60% fewer events than naive 60 Hz on a dense test file.
7. `sound2midi-mpe-play` shows up as a MIDI source in a DAW, streams ≥5 min with <10 ms drift, Ctrl-C leaves nothing hanging; `--dry` vs. full render is audibly different (manual).
8. Loudness: on a stem with a clear crescendo, extracted velocities increase monotonically across the passage; `--vel-blend 0` uses loudness only.
9. Profiles: rendering a `.stems.mid` auto-assigns per-track profiles from stem names; `pluck` tracks contain no pitchwheel vibrato events.
10. Rubato: `--rubato 0` yields grid-exact onsets (bit-identical timing to input); at 0.5, max cumulative displacement ≤ 80 ms and all tracks share the same warp (chord attacks stay within the asynchrony window); fixed seed → reproducible.
11. Legato glide: on a monophonic `sung` track, qualifying transitions render as one sustained note per group with S-curve bend travel; instantaneous bend never leaves [−8192, 8191]; a chain exceeding the bend budget splits with a clean retrigger; `--legato off` restores one note event per source note.
12. Legato overlap: overlapping notes never exceed 15 channels live; successor attack transient measurably reduced.
13. Lint/format/type-check clean; `player` and `sound2midi-mei` untouched and still working.

## 12. Non-goals (v1)

- No player/FluidSynth integration (FluidSynth isn't MPE; possible v2 "MPE out" engine).
- No ML/learned expression — rules + seeded stochastics only.
- No rubato *extracted from the source performance* (v2: since `meter.json` already carries the real beat grid from audio, real human timing could be transferred onto quantized scores — leave a hook in `timing.py` for an external warp map).
- No changes to transcription, artifact generation, notation export, or MEI code paths (only the Harte-parser extraction refactor, kept behavior-identical).
