"""FluidSynth-backed MIDI playback engine with per-track solo/mute.

Parses a MIDI file with ``mido`` into an absolute-time event list (one entry per
channel message, tagged with its source track), then plays it in a background thread,
dispatching each event to a live FluidSynth instance. Solo/mute decisions are made per
event at dispatch time, so tracks can be toggled while playing ("separate vs together").
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fluidsynth
import mido

# General MIDI program names (program number == index).
GM_INSTRUMENTS: tuple[str, ...] = (
    "Acoustic Grand Piano",
    "Bright Acoustic Piano",
    "Electric Grand Piano",
    "Honky-tonk Piano",
    "Electric Piano 1",
    "Electric Piano 2",
    "Harpsichord",
    "Clavinet",
    "Celesta",
    "Glockenspiel",
    "Music Box",
    "Vibraphone",
    "Marimba",
    "Xylophone",
    "Tubular Bells",
    "Dulcimer",
    "Drawbar Organ",
    "Percussive Organ",
    "Rock Organ",
    "Church Organ",
    "Reed Organ",
    "Accordion",
    "Harmonica",
    "Tango Accordion",
    "Acoustic Guitar (nylon)",
    "Acoustic Guitar (steel)",
    "Electric Guitar (jazz)",
    "Electric Guitar (clean)",
    "Electric Guitar (muted)",
    "Overdriven Guitar",
    "Distortion Guitar",
    "Guitar Harmonics",
    "Acoustic Bass",
    "Electric Bass (finger)",
    "Electric Bass (pick)",
    "Fretless Bass",
    "Slap Bass 1",
    "Slap Bass 2",
    "Synth Bass 1",
    "Synth Bass 2",
    "Violin",
    "Viola",
    "Cello",
    "Contrabass",
    "Tremolo Strings",
    "Pizzicato Strings",
    "Orchestral Harp",
    "Timpani",
    "String Ensemble 1",
    "String Ensemble 2",
    "Synth Strings 1",
    "Synth Strings 2",
    "Choir Aahs",
    "Voice Oohs",
    "Synth Voice",
    "Orchestra Hit",
    "Trumpet",
    "Trombone",
    "Tuba",
    "Muted Trumpet",
    "French Horn",
    "Brass Section",
    "Synth Brass 1",
    "Synth Brass 2",
    "Soprano Sax",
    "Alto Sax",
    "Tenor Sax",
    "Baritone Sax",
    "Oboe",
    "English Horn",
    "Bassoon",
    "Clarinet",
    "Piccolo",
    "Flute",
    "Recorder",
    "Pan Flute",
    "Blown Bottle",
    "Shakuhachi",
    "Whistle",
    "Ocarina",
    "Lead 1 (square)",
    "Lead 2 (sawtooth)",
    "Lead 3 (calliope)",
    "Lead 4 (chiff)",
    "Lead 5 (charang)",
    "Lead 6 (voice)",
    "Lead 7 (fifths)",
    "Lead 8 (bass + lead)",
    "Pad 1 (new age)",
    "Pad 2 (warm)",
    "Pad 3 (polysynth)",
    "Pad 4 (choir)",
    "Pad 5 (bowed)",
    "Pad 6 (metallic)",
    "Pad 7 (halo)",
    "Pad 8 (sweep)",
    "FX 1 (rain)",
    "FX 2 (soundtrack)",
    "FX 3 (crystal)",
    "FX 4 (atmosphere)",
    "FX 5 (brightness)",
    "FX 6 (goblins)",
    "FX 7 (echoes)",
    "FX 8 (sci-fi)",
    "Sitar",
    "Banjo",
    "Shamisen",
    "Koto",
    "Kalimba",
    "Bagpipe",
    "Fiddle",
    "Shanai",
    "Tinkle Bell",
    "Agogo",
    "Steel Drums",
    "Woodblock",
    "Taiko Drum",
    "Melodic Tom",
    "Synth Drum",
    "Reverse Cymbal",
    "Guitar Fret Noise",
    "Breath Noise",
    "Seashore",
    "Bird Tweet",
    "Telephone Ring",
    "Helicopter",
    "Applause",
    "Gunshot",
)

# Soundfonts to try, in order, when none is given.
_SOUNDFONT_CANDIDATES = (
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
    "/usr/share/sounds/sf3/MuseScore_General.sf3",
    "/usr/share/sounds/sf3/default-GM.sf3",
    "/usr/share/sounds/sf2/TimGM6mb.sf2",
)

_CHANNEL_MESSAGE_TYPES = frozenset(
    {"note_on", "note_off", "control_change", "program_change", "pitchwheel"}
)

# Synthesized (non-MIDI-file) tracks — e.g. the realized chord accompaniment —
# play on a channel beyond the file's 16. pyfluidsynth allocates 256 MIDI
# channels by default, so channel 16 is always free.
CHORD_CHANNEL = 16


@dataclass
class _SynthMessage:
    """A minimal stand-in for a mido channel message, for synthesized tracks
    (mido validates channel <= 15, so it can't carry the chord channel)."""

    type: str  # "note_on" | "note_off"
    channel: int
    note: int
    velocity: int


def find_soundfont(override: str | None = None) -> Path:
    """Locate a soundfont (.sf2/.sf3). Honors arg, then $SOUND2MIDI_SOUNDFONT, then defaults."""
    for candidate in (override, os.environ.get("SOUND2MIDI_SOUNDFONT"), *_SOUNDFONT_CANDIDATES):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError(
        "No soundfont found. Install one (e.g. `fluid-soundfont-gm`) or set "
        "$SOUND2MIDI_SOUNDFONT to a .sf2/.sf3 file."
    )


@dataclass
class TrackInfo:
    index: int
    name: str
    program: int
    is_drum: bool
    channels: tuple[int, ...]
    note_count: int
    notes: list[tuple[float, float, int]] = field(default_factory=list)  # (start, end, pitch)


@dataclass
class _Event:
    time: float  # absolute seconds
    order: int  # tie-breaker so setup precedes notes at the same instant
    track: int
    msg: Any  # mido.Message (attributes are set dynamically; mido ships no types)


@dataclass
class LoadedSong:
    events: list[_Event]
    tracks: list[TrackInfo]
    duration: float = 0.0
    path: Path | None = field(default=None)


def polyphony_ratio(notes: list[tuple[float, float, int]]) -> float:
    """Fraction of sounding time where 2+ notes overlap. ~0 for a melodic line."""
    if not notes:
        return 0.0
    events = sorted([(start, 1) for start, _, _ in notes] + [(end, -1) for _, end, _ in notes])
    sounding = poly = 0.0
    depth = 0
    prev = events[0][0]
    for t, delta in events:
        span = t - prev
        if depth >= 1:
            sounding += span
        if depth >= 2:
            poly += span
        depth += delta
        prev = t
    return poly / sounding if sounding > 0 else 0.0


def _message_order(msg: Any) -> int:
    return {
        "program_change": 0,
        "control_change": 1,
        "pitchwheel": 2,
        "note_off": 3,
        "note_on": 4,
    }.get(msg.type, 5)


def _tempo_map(mid: mido.MidiFile) -> list[tuple[int, float, int]]:
    """Return [(start_tick, start_seconds, tempo)] segments covering the whole file."""
    changes: list[tuple[int, int]] = []
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                changes.append((tick, msg.tempo))
    changes.sort(key=lambda c: c[0])

    merged: list[tuple[int, int]] = []
    for tick, tempo in changes:
        if merged and merged[-1][0] == tick:
            merged[-1] = (tick, tempo)
        else:
            merged.append((tick, tempo))
    if not merged or merged[0][0] != 0:
        merged.insert(0, (0, 500000))  # default 120 BPM

    segments: list[tuple[int, float, int]] = []
    for i, (tick, tempo) in enumerate(merged):
        if i == 0:
            segments.append((tick, 0.0, tempo))
        else:
            ptick, psec, ptempo = segments[-1]
            sec = psec + mido.tick2second(tick - ptick, mid.ticks_per_beat, ptempo)
            segments.append((tick, sec, tempo))
    return segments


def load_song(path: str | Path) -> LoadedSong:
    """Parse a MIDI file into an absolute-time event list plus per-track metadata."""
    path = Path(path)
    mid = mido.MidiFile(str(path))
    segments = _tempo_map(mid)
    tpb = mid.ticks_per_beat

    def tick_to_sec(tick: int) -> float:
        seg = segments[0]
        for candidate in segments:
            if candidate[0] <= tick:
                seg = candidate
            else:
                break
        return seg[1] + mido.tick2second(tick - seg[0], tpb, seg[2])

    events: list[_Event] = []
    tracks: list[TrackInfo] = []

    for ti, track in enumerate(mid.tracks):
        tick = 0
        channels: set[int] = set()
        program = 0
        name = track.name.strip() if track.name else ""
        active: dict[tuple[int, int], float] = {}
        track_notes: list[tuple[float, float, int]] = []
        last_sec = 0.0
        for msg in track:
            tick += msg.time
            if msg.is_meta:
                continue
            if msg.type in _CHANNEL_MESSAGE_TYPES:
                sec = tick_to_sec(tick)
                last_sec = sec
                channels.add(msg.channel)
                if msg.type == "program_change":
                    program = msg.program
                elif msg.type == "note_on" and msg.velocity > 0:
                    active[(msg.channel, msg.note)] = sec
                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    start = active.pop((msg.channel, msg.note), None)
                    if start is not None:
                        track_notes.append((start, sec, msg.note))
                events.append(_Event(sec, _message_order(msg), ti, msg))

        if not channels:
            continue  # meta-only track (tempo/markers); not a playable instrument
        for (_, note), start in active.items():  # close notes still held at the end
            track_notes.append((start, last_sec, note))
        track_notes.sort(key=lambda n: n[0])
        is_drum = 9 in channels
        if not name:
            name = "Drums" if is_drum else GM_INSTRUMENTS[program % 128]
        tracks.append(
            TrackInfo(
                index=ti,
                name=name,
                program=program,
                is_drum=is_drum,
                channels=tuple(sorted(channels)),
                note_count=len(track_notes),
                notes=track_notes,
            )
        )

    events.sort(key=lambda e: (e.time, e.order))
    duration = max((e.time for e in events), default=0.0)
    return LoadedSong(events=events, tracks=tracks, duration=duration, path=path)


class PlayerEngine:
    """Realtime MIDI playback with per-track solo/mute, backed by FluidSynth."""

    def __init__(
        self,
        *,
        soundfont: str | None = None,
        driver: str | None = None,
        samplerate: int = 44100,
        gain: float = 0.6,
    ) -> None:
        self.soundfont = str(find_soundfont(soundfont))
        self.driver = driver or os.environ.get("SOUND2MIDI_FLUID_DRIVER") or None
        self.samplerate = samplerate
        self.gain = gain

        self._fs: fluidsynth.Synth | None = None
        self._sfid: int | None = None
        self._lock = threading.RLock()

        self.song = LoadedSong(events=[], tracks=[])
        self._muted: set[int] = set()
        self._solo: set[int] = set()

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._playing = False
        self._pos_base = 0.0  # playback start offset (seconds)
        self._wall_start = 0.0  # monotonic() reference for the current run
        self._cursor = 0  # index of next event to fire

        self.on_finished: Callable[[], None] | None = None

    # -- synth lifecycle -------------------------------------------------
    def _ensure_synth(self, *, start_driver: bool = True) -> fluidsynth.Synth:
        if self._fs is None:
            fs = fluidsynth.Synth(samplerate=float(self.samplerate), gain=self.gain)
            if start_driver:
                fs.start(driver=self.driver)
            self._sfid = fs.sfload(self.soundfont)
            self._fs = fs
            self._reset_programs()
        return self._fs

    def _reset_programs(self) -> None:
        fs, sfid = self._fs, self._sfid
        if fs is None or sfid is None:
            return
        for ch in range(16):
            bank = 128 if ch == 9 else 0  # channel 10 (index 9) is GM percussion
            fs.program_select(ch, sfid, bank, 0)
        fs.program_select(CHORD_CHANNEL, sfid, 0, 0)  # synthesized chords: piano

    def close(self) -> None:
        self.stop()
        with self._lock:
            if self._fs is not None:
                self._fs.delete()
                self._fs = None
                self._sfid = None

    # -- track audibility ------------------------------------------------
    def _audible(self, track_index: int) -> bool:
        if self._solo:
            return track_index in self._solo
        return track_index not in self._muted

    def set_muted(self, track_index: int, muted: bool) -> None:
        with self._lock:
            if muted:
                self._muted.add(track_index)
            else:
                self._muted.discard(track_index)
        self._apply_audibility()

    def set_solo(self, track_index: int, solo: bool) -> None:
        with self._lock:
            if solo:
                self._solo.add(track_index)
            else:
                self._solo.discard(track_index)
        self._apply_audibility()

    def clear_solo(self) -> None:
        with self._lock:
            self._solo.clear()
        self._apply_audibility()

    def unmute_all(self) -> None:
        with self._lock:
            self._muted.clear()
        self._apply_audibility()

    def is_muted(self, track_index: int) -> bool:
        return track_index in self._muted

    def is_solo(self, track_index: int) -> bool:
        return track_index in self._solo

    def is_audible(self, track_index: int) -> bool:
        return self._audible(track_index)

    def any_solo(self) -> bool:
        return bool(self._solo)

    def _apply_audibility(self) -> None:
        """Silence any now-inaudible tracks immediately (kills sustained notes)."""
        with self._lock:
            if self._fs is None:
                return
            for track in self.song.tracks:
                if not self._audible(track.index):
                    for ch in track.channels:
                        self._fs.cc(ch, 123, 0)  # all notes off

    def set_gain(self, gain: float) -> None:
        self.gain = gain
        with self._lock:
            if self._fs is not None:
                self._fs.setting("synth.gain", float(gain))

    # -- loading ---------------------------------------------------------
    def load(self, path: str | Path) -> LoadedSong:
        self.stop()
        with self._lock:
            self.song = load_song(path)
            self._muted.clear()
            self._solo.clear()
            self._pos_base = 0.0
            self._cursor = 0
        return self.song

    def _chord_events(
        self, index: int, chord_notes: list[tuple[float, float, tuple[int, ...]]]
    ) -> tuple[list[_Event], list[tuple[float, float, int]]]:
        """Build the note events and lane notes for a synthesized chord track.

        The first note of each chord (the bass) is voiced slightly louder.
        """
        events: list[_Event] = []
        track_notes: list[tuple[float, float, int]] = []
        for start, end, pitches in chord_notes:
            for i, note_num in enumerate(pitches):
                velocity = 78 if i == 0 else 58
                on = _SynthMessage("note_on", CHORD_CHANNEL, note_num, velocity)
                off = _SynthMessage("note_off", CHORD_CHANNEL, note_num, 0)
                events.append(_Event(start, _message_order(on), index, on))
                events.append(_Event(end, _message_order(off), index, off))
                track_notes.append((start, end, note_num))
        return events, sorted(track_notes)

    def add_chord_track(
        self,
        chord_notes: list[tuple[float, float, tuple[int, ...]]],
        *,
        name: str = "Chords (piano)",
    ) -> TrackInfo:
        """Append a synthesized track playing ``(start, end, midi_notes)`` chords.

        The track behaves like any loaded track (solo/mute, seek, offline render)
        but lives on :data:`CHORD_CHANNEL`, beyond the MIDI file's 16 channels.
        """
        with self._lock:
            index = max((t.index for t in self.song.tracks), default=-1) + 1
            events, track_notes = self._chord_events(index, chord_notes)
            self.song.events.extend(events)
            self.song.events.sort(key=lambda e: (e.time, e.order))
            info = TrackInfo(
                index=index,
                name=name,
                program=0,
                is_drum=False,
                channels=(CHORD_CHANNEL,),
                note_count=len(track_notes),
                notes=track_notes,
            )
            self.song.tracks.append(info)
        return info

    def update_chord_track(
        self, index: int, chord_notes: list[tuple[float, float, tuple[int, ...]]]
    ) -> None:
        """Replace a synthesized chord track's notes (e.g. on a style change).

        Keeps the track index — and with it the solo/mute state and lane wiring —
        and resumes playback from the current position if it was playing.
        """
        was_playing = self._playing
        position = self.position()
        self._halt_thread()
        with self._lock:
            self.song.events = [e for e in self.song.events if e.track != index]
            events, track_notes = self._chord_events(index, chord_notes)
            self.song.events.extend(events)
            self.song.events.sort(key=lambda e: (e.time, e.order))
            for info in self.song.tracks:
                if info.index == index:
                    info.note_count = len(track_notes)
                    info.notes = track_notes
                    break
        self._pos_base = position
        self._cursor = self._index_at(position)
        if was_playing:
            self.play()

    # -- dispatch --------------------------------------------------------
    def _dispatch(self, event: _Event) -> None:
        fs, sfid = self._fs, self._sfid
        if fs is None or sfid is None:
            return
        msg = event.msg
        ch = msg.channel
        if msg.type == "program_change":
            bank = 128 if ch == 9 else 0
            fs.program_select(ch, sfid, bank, msg.program)
        elif msg.type == "control_change":
            fs.cc(ch, msg.control, msg.value)
        elif msg.type == "pitchwheel":
            fs.pitch_bend(ch, msg.pitch)
        elif msg.type == "note_on" and msg.velocity > 0:
            if self._audible(event.track):
                fs.noteon(ch, msg.note, msg.velocity)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            fs.noteoff(ch, msg.note)

    def _apply_state_until(self, cursor: int) -> None:
        """Replay program/control/pitch (not notes) before ``cursor`` after a seek."""
        with self._lock:
            for event in self.song.events[:cursor]:
                if event.msg.type in ("program_change", "control_change", "pitchwheel"):
                    self._dispatch(event)

    # -- transport -------------------------------------------------------
    @property
    def duration(self) -> float:
        return self.song.duration

    @property
    def playing(self) -> bool:
        return self._playing

    def position(self) -> float:
        if self._playing:
            return min(time.monotonic() - self._wall_start, self.song.duration)
        return self._pos_base

    def play(self) -> None:
        if self._playing or not self.song.events:
            return
        self._ensure_synth()
        self._reset_programs()
        self._apply_state_until(self._cursor)
        self._stop.clear()
        self._wall_start = time.monotonic() - self._pos_base
        self._playing = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        events = self.song.events
        n = len(events)
        while self._cursor < n and not self._stop.is_set():
            event = events[self._cursor]
            now = time.monotonic() - self._wall_start
            delay = event.time - now
            if delay > 0:
                self._stop.wait(min(delay, 0.01))
                continue
            with self._lock:
                self._dispatch(event)
            self._cursor += 1

        finished_naturally = self._cursor >= n and not self._stop.is_set()
        self._playing = False
        self._all_sound_off()
        if finished_naturally:
            self._pos_base = 0.0
            self._cursor = 0
            if self.on_finished is not None:
                self.on_finished()

    def pause(self) -> None:
        if not self._playing:
            return
        pos = self.position()
        self._halt_thread()
        self._pos_base = pos

    def stop(self) -> None:
        self._halt_thread()
        self._pos_base = 0.0
        self._cursor = 0

    def _halt_thread(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None
        self._playing = False
        self._all_sound_off()

    def _all_sound_off(self) -> None:
        with self._lock:
            if self._fs is None:
                return
            for ch in (*range(16), CHORD_CHANNEL):
                self._fs.cc(ch, 123, 0)
                self._fs.cc(ch, 120, 0)

    def seek(self, seconds: float) -> None:
        seconds = max(0.0, min(seconds, self.song.duration))
        was_playing = self._playing
        self._halt_thread()
        self._pos_base = seconds
        self._cursor = self._index_at(seconds)
        if was_playing:
            self.play()

    def _index_at(self, seconds: float) -> int:
        lo, hi = 0, len(self.song.events)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.song.events[mid].time < seconds:
                lo = mid + 1
            else:
                hi = mid
        return lo

    # -- offline render --------------------------------------------------
    def render_wav(self, out_path: str | Path, *, duration: float | None = None) -> Path:
        """Render the current song (honoring solo/mute) to a stereo 16-bit WAV.

        Uses FluidSynth without an audio driver, so it works headless and is the
        basis for the engine's smoke test.
        """
        import wave

        import numpy as np

        out_path = Path(out_path)
        total = self.song.duration if duration is None else min(duration, self.song.duration)
        fs = fluidsynth.Synth(samplerate=float(self.samplerate), gain=self.gain)
        sfid = fs.sfload(self.soundfont)
        # Temporarily bind so _dispatch/_reset use this offline synth.
        prev_fs, prev_sfid = self._fs, self._sfid
        self._fs, self._sfid = fs, sfid
        try:
            self._reset_programs()
            chunks: list = []
            t = 0.0
            for event in self.song.events:
                if event.time > total:
                    break
                gap = event.time - t
                if gap > 0:
                    chunks.append(fs.get_samples(int(gap * self.samplerate)))
                    t = event.time
                self._dispatch(event)
            tail = total - t
            if tail > 0:
                chunks.append(fs.get_samples(int(tail * self.samplerate)))
            samples = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        finally:
            fs.delete()
            self._fs, self._sfid = prev_fs, prev_sfid

        with wave.open(str(out_path), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(self.samplerate)
            wav.writeframes(samples.astype("<i2").tobytes())
        return out_path
