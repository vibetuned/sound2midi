"""PySide6 window for the MIDI player. Imported lazily by ``app.main``."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from sound2midi.player import chordlabel
from sound2midi.player.chordstrip import ChordLane, build_chords
from sound2midi.player.engine import PlayerEngine, polyphony_ratio
from sound2midi.player.export import export_to_staff
from sound2midi.player.pianoroll import InstrumentLane
from sound2midi.player.sectionstrip import Section, SectionLane, build_sections

# When looping a section, wrap if the playhead is within this many seconds past
# its end (the tick timer samples every 80 ms; a seek can overshoot further).
_LOOP_GRACE = 1.0


def _fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


class PlayerWindow(QMainWindow):
    finished = Signal()
    export_done = Signal(object)  # dict[str, Path] on success, Exception on failure

    def __init__(self, *, soundfont: str | None = None, driver: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("sound2midi player")
        self.engine = PlayerEngine(soundfont=soundfont, driver=driver)
        self.engine.on_finished = self.finished.emit  # emitted from playback thread
        self.finished.connect(self._on_finished)
        self.export_done.connect(self._on_export_done)

        self._dragging = False
        self._lanes: list[InstrumentLane] = []
        self._sections: list[Section] = []
        self._section_lane: SectionLane | None = None
        self._selected_sections: list[int] = []  # strip selection (section indices)
        self._loops: list[tuple[float, float]] = []  # playback windows, in song order
        self._chords: list[dict] = []  # raw chord segments for export
        self._chord_lane: ChordLane | None = None
        self._detected_key: str | None = None
        self._detected_meter: dict | None = None
        self._build_ui()

        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # -- UI construction -------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        file_row = QHBoxLayout()
        open_btn = QPushButton("Open MIDI…")
        open_btn.clicked.connect(self._open_dialog)
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("color: #888;")
        file_row.addWidget(open_btn)
        file_row.addWidget(self.file_label, 1)
        root.addLayout(file_row)

        transport = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        self.play_btn.setEnabled(False)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        self.position = QSlider(Qt.Orientation.Horizontal)
        self.position.setRange(0, 1000)
        self.position.setEnabled(False)
        self.position.sliderPressed.connect(self._on_seek_start)
        self.position.sliderReleased.connect(self._on_seek_end)
        self.time_label = QLabel("0:00 / 0:00")
        transport.addWidget(self.play_btn)
        transport.addWidget(self.stop_btn)
        transport.addWidget(self.position, 1)
        transport.addWidget(self.time_label)
        transport.addSpacing(12)
        transport.addWidget(QLabel("Volume"))
        self.gain = QSlider(Qt.Orientation.Horizontal)
        self.gain.setRange(0, 150)
        self.gain.setValue(int(self.engine.gain * 100))
        self.gain.setMaximumWidth(160)
        self.gain.valueChanged.connect(lambda v: self.engine.set_gain(v / 100.0))
        transport.addWidget(self.gain)
        clear_solo = QPushButton("Clear solo")
        clear_solo.clicked.connect(self._clear_solo)
        unmute = QPushButton("Unmute all")
        unmute.clicked.connect(self._unmute_all)
        transport.addWidget(clear_solo)
        transport.addWidget(unmute)
        root.addLayout(transport)

        # Export controls, split over two rows: how the score is built, then
        # what goes into it and the output formats.
        export1 = QHBoxLayout()
        export1.addWidget(QLabel("Export staff:"))
        self.staff_mode = QComboBox()
        self.staff_mode.addItem("Single staff", "single")
        self.staff_mode.addItem("Grand staff (2)", "grand")
        self.staff_mode.addItem("Split (auto)", "split")
        self.staff_mode.setToolTip(
            "Single: everything on one staff. Grand: you assign instruments to the "
            "two staves. Split: one selection, split into treble/bass automatically "
            "around a moving middle that follows the notes' pitch, bar by bar."
        )
        self.staff_mode.currentIndexChanged.connect(self._sync_export_controls)
        export1.addWidget(self.staff_mode)
        self.staff_hint = QLabel("tick ① to include")
        self.staff_hint.setStyleSheet("color: #888;")
        export1.addWidget(self.staff_hint)
        export1.addWidget(QLabel("Grid:"))
        self.grid_combo = QComboBox()
        self.grid_combo.setToolTip("Notation quantization grid (coarser = cleaner, fewer tuplets)")
        self.grid_combo.addItem("1/4", (1,))
        self.grid_combo.addItem("1/8", (2,))
        self.grid_combo.addItem("1/16", (4,))
        self.grid_combo.addItem("1/32", (8,))
        self.grid_combo.addItem("1/8 + triplets", (2, 3))
        self.grid_combo.addItem("1/16 + triplets", (4, 3))
        self.grid_combo.setCurrentIndex(2)  # 1/16 grid: clean default
        export1.addWidget(self.grid_combo)
        self.trim_box = QCheckBox("Legato trim")
        self.trim_box.setChecked(True)
        self.trim_box.setToolTip(
            "Cut notes that ring slightly past the next onset, so legato lines don't "
            "turn into sliver chords (A → C becoming A/[A C]/C). Real chords and long "
            "suspensions are kept."
        )
        export1.addWidget(self.trim_box)
        export1.addStretch(1)
        root.addLayout(export1)

        export2 = QHBoxLayout()
        self.apply_key_box = QCheckBox("Key: —")
        self.apply_key_box.setEnabled(False)
        self.apply_key_box.setToolTip(
            "Apply the detected key signature (from <song>.key.json) to the exported score"
        )
        export2.addWidget(self.apply_key_box)
        self.apply_meter_box = QCheckBox("Meter: —")
        self.apply_meter_box.setEnabled(False)
        self.apply_meter_box.setToolTip(
            "Retime the export to the detected beat grid (from <song>.meter.json): real "
            "tempo + time signature, bars anchored on downbeats"
        )
        export2.addWidget(self.apply_meter_box)
        self.apply_chords_box = QCheckBox("Chords")
        self.apply_chords_box.setEnabled(False)
        self.apply_chords_box.setToolTip(
            "Write the detected chord symbols (from <song>.chords.json) above the top "
            "staff, snapped to the beat grid and respelled to the key"
        )
        export2.addWidget(self.apply_chords_box)
        self.fmt_musicxml = QCheckBox("MusicXML")
        self.fmt_musicxml.setChecked(True)
        self.fmt_abc = QCheckBox("ABC")
        self.fmt_abc.setChecked(True)
        export2.addWidget(self.fmt_musicxml)
        export2.addWidget(self.fmt_abc)
        export2.addStretch(1)
        self.export_btn = QPushButton("Export selected →")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export)
        export2.addWidget(self.export_btn)
        root.addLayout(export2)
        self._sync_export_controls()

        self.lane_area = QScrollArea()
        self.lane_area.setWidgetResizable(True)
        self.lane_host = QWidget()
        self.lane_layout = QVBoxLayout(self.lane_host)
        self.lane_layout.setSpacing(1)
        self.lane_layout.addStretch(1)
        self.lane_area.setWidget(self.lane_host)
        root.addWidget(self.lane_area, 1)

        self.resize(960, 600)

    # -- loading ---------------------------------------------------------
    def load(self, path: str | Path) -> None:
        path = Path(path)
        try:
            song = self.engine.load(path)
        except Exception as exc:  # surface any load failure to the user
            QMessageBox.critical(self, "Failed to load MIDI", f"{path}\n\n{exc}")
            return

        self.file_label.setText(
            f"{path.name}  ·  {len(song.tracks)} tracks  ·  {_fmt_time(song.duration)}"
        )
        self.file_label.setStyleSheet("")
        self.setWindowTitle(f"sound2midi player — {path.name}")

        # Sections (from <song>.sections.json, written by --sections) come first:
        # the lanes need the boundaries for their per-section export cells.
        sections_data = self._load_artifact(path, "sections")
        segments = sections_data.get("segments") if sections_data else None
        self._sections = build_sections(segments) if isinstance(segments, list) else []
        self._selected_sections = []
        self._loops = []
        section_spans = [(s.start, s.end) for s in self._sections]

        for lane in self._lanes:
            lane.setParent(None)
            lane.deleteLater()
        self._lanes.clear()
        n = len(song.tracks)
        for track in song.tracks:
            # Melodic tracks sit well below ~0.3 polyphony (their overlap is mostly
            # transcription ring); real polyphony (piano, guitars, pads) sits above.
            mono_default = (
                not track.is_drum and track.note_count >= 10 and polyphony_ratio(track.notes) < 0.30
            )
            lane = InstrumentLane(
                track,
                self.engine,
                n,
                on_seek=self._after_seek,
                on_toggle=self._refresh_lanes,
                mono_default=mono_default,
                sections=section_spans,
            )
            self._lanes.append(lane)
            self.lane_layout.insertWidget(self.lane_layout.count() - 1, lane)

        if self._section_lane is not None:
            self._section_lane.setParent(None)
            self._section_lane.deleteLater()
            self._section_lane = None
        if self._sections:
            self._section_lane = SectionLane(
                self._sections,
                self.engine,
                on_seek=self._after_seek,
                on_selection=self._on_sections_selected,
            )
            self.lane_layout.insertWidget(0, self._section_lane)

        # Key and meter artifacts load before the chord strip: the arpeggio
        # realization wants the beat grid.
        self._detected_key = self._load_key(path)
        self._detected_meter = self._load_meter(path)

        # Chord strip (from <song>.chords.json, written by --chords).
        if self._chord_lane is not None:
            self._chord_lane.setParent(None)
            self._chord_lane.deleteLater()
            self._chord_lane = None
        chords_data = self._load_artifact(path, "chords")
        segments = chords_data.get("chords") if chords_data else None
        self._chords = segments if isinstance(segments, list) else []
        if self._chords:
            # Realize the progression as a synthesized piano track so it can be
            # auditioned (muted by default, Solo/Mute on the lane) and exported
            # as a staff. Chord times are seconds, the engine's own clock.
            chord_notes = self._realize_chord_notes("block")
            chord_track = None
            if chord_notes:
                chord_track = self.engine.add_chord_track(chord_notes).index
                self.engine.set_muted(chord_track, True)  # opt-in listening
            self._chord_lane = ChordLane(
                build_chords(self._chords),
                self.engine,
                track_index=chord_track,
                sections=section_spans,
                on_seek=self._after_seek,
                on_toggle=self._refresh_lanes,
                on_style=self._on_chord_style,
            )
            self.lane_layout.insertWidget(1 if self._section_lane else 0, self._chord_lane)
        self.apply_chords_box.setEnabled(bool(self._chords))
        self.apply_chords_box.setChecked(bool(self._chords))
        self._sync_export_controls()  # grand-mode staff boxes on the new lanes
        if self._detected_key:
            self.apply_key_box.setText(f"Key: {self._detected_key}")
            self.apply_key_box.setEnabled(True)
            self.apply_key_box.setChecked(True)
        else:
            self.apply_key_box.setText("Key: —")
            self.apply_key_box.setChecked(False)
            self.apply_key_box.setEnabled(False)

        if self._detected_meter:
            ts = self._detected_meter.get("time_signature", "?")
            bpm = self._detected_meter.get("bpm")
            label = f"Meter: {ts}" + (f" ♩≈{round(bpm)}" if bpm else "")
            self.apply_meter_box.setText(label)
            self.apply_meter_box.setEnabled(True)
            self.apply_meter_box.setChecked(True)
        else:
            self.apply_meter_box.setText("Meter: —")
            self.apply_meter_box.setChecked(False)
            self.apply_meter_box.setEnabled(False)

        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.position.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.position.setValue(0)
        self.play_btn.setText("Play")
        self._update_time()

    @staticmethod
    def _load_artifact(midi_path: Path, suffix: str) -> dict | None:
        """Find and read the song's <song>.<suffix>.json artifact, if any.

        Artifacts live in the song folder's ``artifacts/`` subfolder; the song
        folder root is still searched for folders from before that layout.
        """
        parent = midi_path.parent
        stem = midi_path.stem
        candidates = []
        for folder in (parent / "artifacts", parent):
            candidates += [
                folder / f"{parent.name}.{suffix}.json",
                folder / f"{stem}.{suffix}.json",
                folder / f"{stem.replace('.stems', '')}.{suffix}.json",
                *sorted(folder.glob(f"*.{suffix}.json")),
            ]
        for path in candidates:
            if path.is_file():
                try:
                    data = json.loads(path.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                if isinstance(data, dict):
                    return data
        return None

    def _load_key(self, midi_path: Path) -> str | None:
        data = self._load_artifact(midi_path, "key")
        return data.get("key") if data else None

    def _load_meter(self, midi_path: Path) -> dict | None:
        data = self._load_artifact(midi_path, "meter")
        return data if data and data.get("time_signature") else None

    def _open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open MIDI file", "", "MIDI files (*.mid *.midi);;All files (*)"
        )
        if path:
            self.load(path)

    # -- transport slots -------------------------------------------------
    def _toggle_play(self) -> None:
        if self.engine.playing:
            self.engine.pause()
            self.play_btn.setText("Play")
        else:
            self.engine.play()
            self.play_btn.setText("Pause")

    def _stop(self) -> None:
        self.engine.stop()
        self.play_btn.setText("Play")
        self.position.setValue(0)
        self._refresh_playheads()
        self._update_time()

    def _on_finished(self) -> None:
        if self._loops and self.engine.duration > 0:
            # A looped final section runs into the end of the song; wrap around.
            self.engine.seek(self._loops[0][0])
            self.engine.play()
            self.play_btn.setText("Pause")
            self._refresh_playheads()
            return
        self.play_btn.setText("Play")
        self.position.setValue(0)
        self._refresh_playheads()
        self._update_time()

    def _on_seek_start(self) -> None:
        self._dragging = True

    def _on_seek_end(self) -> None:
        if self.engine.duration > 0:
            self.engine.seek(self.position.value() / 1000.0 * self.engine.duration)
        self._dragging = False
        if self.engine.playing:
            self.play_btn.setText("Pause")
        self._refresh_playheads()

    def _after_seek(self) -> None:
        """Called when a piano-roll lane is clicked to seek."""
        self._update_time()
        if self.engine.playing:
            self.play_btn.setText("Pause")

    def _realize_chord_notes(self, style: str) -> list[tuple[float, float, tuple[int, ...]]]:
        """Realize the chords artifact as notes in the given style, clamped to the
        song and using the detected beat grid (for arpeggios) when present."""
        duration = self.engine.duration
        segs = []
        for seg in build_chords(self._chords):
            end = min(seg.end, duration)
            if end - seg.start >= 0.05:
                segs.append((seg.label, seg.start, end))
        beats = None
        if self._detected_meter:
            beats = [float(b) for b in self._detected_meter.get("beats") or []] or None
        return chordlabel.realize_chords(segs, style, beats=beats)

    def _on_chord_style(self, style: str) -> None:
        """Rebuild the synthesized chord track when the lane's style changes."""
        if self._chord_lane is None or self._chord_lane.track_index is None:
            return
        self.engine.update_chord_track(
            self._chord_lane.track_index, self._realize_chord_notes(style)
        )
        self._refresh_playheads()

    def _on_sections_selected(self, indices: list[int]) -> None:
        self._selected_sections = list(indices)
        duration = self.engine.duration
        loops = []
        for i in indices:
            section = self._sections[i]
            # A section entirely past the MIDI's end (the audio can outlast the
            # transcription) has nothing to loop.
            if section.start < duration:
                loops.append((section.start, min(section.end, duration)))
        self._loops = loops

    def _clear_solo(self) -> None:
        self.engine.clear_solo()
        self._refresh_lanes()

    def _unmute_all(self) -> None:
        self.engine.unmute_all()
        self._refresh_lanes()

    def _refresh_lanes(self) -> None:
        for lane in self._lanes:
            lane.sync()
        if self._chord_lane is not None:
            self._chord_lane.sync()

    # -- export ----------------------------------------------------------
    def _sync_export_controls(self) -> None:
        mode = str(self.staff_mode.currentData())
        grand = mode == "grand"
        if grand:
            hint = "tick 1 (treble) / 2 (bass) · cells follow the 2-box, else staff 1"
        elif mode == "split":
            hint = "tick 1 to include — notes auto-split into treble/bass"
        else:
            hint = "tick 1 to include"
        self.staff_hint.setText(hint)
        for lane in self._lanes:
            lane.set_grand_mode(grand)
        if self._chord_lane is not None:
            self._chord_lane.set_grand_mode(grand)

    def _export(self) -> None:
        midi = self.engine.song.path
        if midi is None:
            return
        mode = str(self.staff_mode.currentData())  # "single" | "grand" | "split"
        grand = mode == "grand"
        # A track is exported if a staff box is ticked — or if it has section
        # cells (Ctrl+click on its lane), which put it on staff 1 unless it is
        # explicitly assigned to staff 2.
        cell_map = {lane.track.index: lane.cells() for lane in self._lanes}
        if grand:
            staves = [
                [
                    lane.track.index
                    for lane in self._lanes
                    if lane.on_staff1() or (cell_map[lane.track.index] and not lane.on_staff2())
                ],
                [lane.track.index for lane in self._lanes if lane.on_staff2()],
            ]
        else:
            staves = [
                [
                    lane.track.index
                    for lane in self._lanes
                    if lane.on_staff1() or cell_map[lane.track.index]
                ]
            ]
        # The realized chord track exports as a regular voice: its lane has the
        # same staff checkboxes and section cells, and its notes are synthesized
        # (not in the file).
        synth_tracks_arg = None
        chord_lane = self._chord_lane
        chord_track = chord_lane.track_index if chord_lane is not None else None
        chord_cells: set[int] = set()
        if chord_lane is not None and chord_track is not None:
            chord_cells = chord_lane.cells()
            cell_map[chord_track] = chord_cells
            if chord_lane.on_staff1() or chord_lane.on_staff2() or chord_cells:
                synth_tracks_arg = {
                    chord_track: self._realize_chord_notes(chord_lane.chord_style())
                }
                if grand:
                    if chord_lane.on_staff1() or (chord_cells and not chord_lane.on_staff2()):
                        staves[0].append(chord_track)
                    if chord_lane.on_staff2():
                        staves[1].append(chord_track)
                else:
                    staves[0].append(chord_track)

        if not any(staves):
            QMessageBox.information(
                self,
                "Export",
                "Tick a staff checkbox on one or more instruments (or the Chords lane) "
                "first — or Ctrl+click sections on their lanes.",
            )
            return

        # Export window = the union of the strip-selected sections and every
        # section that has cells (whole song when both are empty). A section's
        # cells pick its instruments; a section without cells uses the
        # staff-ticked defaults. The strip selection is read straight off the
        # widget so the export can never disagree with what is on screen.
        strip_selected = (
            set(self._section_lane.strip.selected) if self._section_lane is not None else set()
        )
        section_indices = sorted(strip_selected | {i for cells in cell_map.values() for i in cells})
        sections_arg = [
            {
                "start": self._sections[i].start,
                "end": self._sections[i].end,
                "tracks": ({t for t, cells in cell_map.items() if i in cells} or None),
            }
            for i in section_indices
        ] or None
        if sections_arg and synth_tracks_arg and chord_track is not None and not chord_cells:
            # Chord voice staffed but without cells of its own: keep it in every
            # cell-restricted window. (With cells, it follows them like any track.)
            for window in sections_arg:
                tracks = window["tracks"]
                if isinstance(tracks, set):
                    window["tracks"] = tracks | {chord_track}

        formats = []
        if self.fmt_musicxml.isChecked():
            formats.append("musicxml")
        if self.fmt_abc.isChecked():
            formats.append("abc")
        if not formats:
            QMessageBox.information(self, "Export", "Choose at least one format (MusicXML/ABC).")
            return

        midi_path = Path(midi)
        out_dir = midi_path.parent
        basename = f"{midi_path.stem}.{mode}" + (".sections" if sections_arg else "")
        quantize_divisors = self.grid_combo.currentData()
        formats_tuple = tuple(formats)
        apply_key = self.apply_key_box.isChecked() and self._detected_key
        key = self._detected_key if apply_key else None
        apply_meter = self.apply_meter_box.isChecked() and self._detected_meter
        meter = self._detected_meter if apply_meter else None
        trim_overlaps = self.trim_box.isChecked()
        mono_tracks = frozenset(lane.track.index for lane in self._lanes if lane.is_mono())
        chords_arg = list(self._chords) if self.apply_chords_box.isChecked() else None

        # Human summary of exactly what goes into this export, shown in the
        # completion dialog — so the exported state is never a mystery.
        names = {lane.track.index: lane.track.name for lane in self._lanes}
        if chord_track is not None:
            names[chord_track] = "Chords"
        if sections_arg:
            lines = []
            for i, window in zip(section_indices, sections_arg, strict=True):
                tracks = window["tracks"]
                who = (
                    ", ".join(names.get(t, str(t)) for t in sorted(tracks))
                    if isinstance(tracks, set)
                    else "all selected instruments"
                )
                lines.append(f"{self._sections[i].display}: {who}")
            summary = "\n".join(lines)
        else:
            staffed = sorted({t for staff in staves for t in staff})
            summary = "Whole song: " + ", ".join(names.get(t, str(t)) for t in staffed)

        self.export_btn.setEnabled(False)
        self.export_btn.setText("Exporting…")

        def worker() -> None:
            try:
                result = export_to_staff(
                    midi_path,
                    staves,
                    out_dir,
                    basename=basename,
                    formats=formats_tuple,
                    quantize_divisors=quantize_divisors,
                    key=key,
                    meter=meter,
                    trim_overlaps=trim_overlaps,
                    mono_tracks=mono_tracks,
                    title=midi_path.stem,
                    sections=sections_arg,
                    chords=chords_arg,
                    synth_tracks=synth_tracks_arg,
                    split=(mode == "split"),
                )
                self.export_done.emit((result, summary))
            except Exception as exc:  # delivered to the GUI thread via the signal
                self.export_done.emit(exc)

        threading.Thread(target=worker, daemon=True).start()

    def _on_export_done(self, result: object) -> None:
        self.export_btn.setEnabled(True)
        self.export_btn.setText("Export selected →")
        if isinstance(result, Exception):
            QMessageBox.critical(self, "Export failed", str(result))
            return
        assert isinstance(result, tuple)
        files, summary = result
        assert isinstance(files, dict) and isinstance(summary, str)
        lines = []
        for key, path in files.items():
            label = "ABC (skipped)" if key == "abc_error" else key
            lines.append(f"{label}: {path}")
        QMessageBox.information(
            self, "Export complete", "\n".join(lines) + "\n\nExported:\n" + summary
        )

    def _refresh_playheads(self) -> None:
        for lane in self._lanes:
            lane.refresh_playhead()
        if self._section_lane is not None:
            self._section_lane.refresh_playhead()
        if self._chord_lane is not None:
            self._chord_lane.refresh_playhead()

    # -- periodic update -------------------------------------------------
    def _tick(self) -> None:
        if self._loops and self.engine.playing:
            pos = self.engine.position()
            # Play the selected sections in song order: crossing out of one jumps
            # to the next (wrapping to the first). Only a crossing triggers — a
            # deliberate seek far outside (beyond the grace window) is left alone.
            inside = any(start <= pos < end for start, end in self._loops)
            crossed = any(end <= pos < end + _LOOP_GRACE for _, end in self._loops)
            if not inside and crossed:
                target = min(
                    (start for start, _ in self._loops if start > pos),
                    default=self._loops[0][0],
                )
                self.engine.seek(target)
                self._refresh_playheads()
        if not self._dragging and self.engine.duration > 0:
            frac = self.engine.position() / self.engine.duration
            self.position.setValue(int(frac * 1000))
        if self.engine.playing:
            self._refresh_playheads()
        elif self.play_btn.text() == "Pause":
            self.play_btn.setText("Play")
        self._update_time()

    def _update_time(self) -> None:
        self.time_label.setText(
            f"{_fmt_time(self.engine.position())} / {_fmt_time(self.engine.duration)}"
        )

    def closeEvent(self, event) -> None:  # Qt override name (camelCase required)
        self.engine.close()
        super().closeEvent(event)


def run(*, midi: str | None = None, soundfont: str | None = None, driver: str | None = None) -> int:
    app = QApplication.instance() or QApplication([])
    try:
        window = PlayerWindow(soundfont=soundfont, driver=driver)
    except FileNotFoundError as exc:
        QMessageBox.critical(None, "sound2midi player", str(exc))
        return 1
    window.show()
    if midi and Path(midi).is_file():
        window.load(midi)
    return app.exec()
