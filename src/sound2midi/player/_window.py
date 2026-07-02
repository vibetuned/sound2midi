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

from sound2midi.player.engine import PlayerEngine
from sound2midi.player.export import export_to_staff
from sound2midi.player.pianoroll import InstrumentLane


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
        self._detected_key: str | None = None
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
        root.addLayout(transport)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Volume"))
        self.gain = QSlider(Qt.Orientation.Horizontal)
        self.gain.setRange(0, 150)
        self.gain.setValue(int(self.engine.gain * 100))
        self.gain.setMaximumWidth(160)
        self.gain.valueChanged.connect(lambda v: self.engine.set_gain(v / 100.0))
        controls.addWidget(self.gain)
        controls.addStretch(1)
        clear_solo = QPushButton("Clear solo")
        clear_solo.clicked.connect(self._clear_solo)
        unmute = QPushButton("Unmute all")
        unmute.clicked.connect(self._unmute_all)
        controls.addWidget(clear_solo)
        controls.addWidget(unmute)
        root.addLayout(controls)

        export = QHBoxLayout()
        export.addWidget(QLabel("Export staff:"))
        self.staff_mode = QComboBox()
        self.staff_mode.addItem("Single staff", "single")
        self.staff_mode.addItem("Grand staff (2)", "grand")
        self.staff_mode.currentIndexChanged.connect(self._sync_export_controls)
        export.addWidget(self.staff_mode)
        self.staff_hint = QLabel("tick ① to include")
        self.staff_hint.setStyleSheet("color: #888;")
        export.addWidget(self.staff_hint)
        export.addWidget(QLabel("Grid:"))
        self.grid_combo = QComboBox()
        self.grid_combo.setToolTip("Notation quantization grid (coarser = cleaner, fewer tuplets)")
        self.grid_combo.addItem("1/4", (1,))
        self.grid_combo.addItem("1/8", (2,))
        self.grid_combo.addItem("1/16", (4,))
        self.grid_combo.addItem("1/32", (8,))
        self.grid_combo.addItem("1/8 + triplets", (2, 3))
        self.grid_combo.addItem("1/16 + triplets", (4, 3))
        self.grid_combo.setCurrentIndex(2)  # 1/16 grid: clean default
        export.addWidget(self.grid_combo)
        self.apply_key_box = QCheckBox("Key: —")
        self.apply_key_box.setEnabled(False)
        self.apply_key_box.setToolTip(
            "Apply the detected key signature (from <song>.key.json) to the exported score"
        )
        export.addWidget(self.apply_key_box)
        self.fmt_musicxml = QCheckBox("MusicXML")
        self.fmt_musicxml.setChecked(True)
        self.fmt_abc = QCheckBox("ABC")
        self.fmt_abc.setChecked(True)
        export.addWidget(self.fmt_musicxml)
        export.addWidget(self.fmt_abc)
        export.addStretch(1)
        self.export_btn = QPushButton("Export selected →")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export)
        export.addWidget(self.export_btn)
        root.addLayout(export)
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

        for lane in self._lanes:
            lane.setParent(None)
            lane.deleteLater()
        self._lanes.clear()
        n = len(song.tracks)
        for track in song.tracks:
            lane = InstrumentLane(
                track,
                self.engine,
                n,
                on_seek=self._after_seek,
                on_toggle=self._refresh_lanes,
            )
            self._lanes.append(lane)
            self.lane_layout.insertWidget(self.lane_layout.count() - 1, lane)

        self._detected_key = self._load_key(path)
        if self._detected_key:
            self.apply_key_box.setText(f"Key: {self._detected_key}")
            self.apply_key_box.setEnabled(True)
            self.apply_key_box.setChecked(True)
        else:
            self.apply_key_box.setText("Key: —")
            self.apply_key_box.setChecked(False)
            self.apply_key_box.setEnabled(False)

        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.position.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.position.setValue(0)
        self.play_btn.setText("Play")
        self._update_time()

    @staticmethod
    def _load_key(midi_path: Path) -> str | None:
        """Find and read a sibling <song>.key.json artifact, if any."""
        parent = midi_path.parent
        stem = midi_path.stem
        candidates = [
            parent / f"{parent.name}.key.json",
            parent / f"{stem}.key.json",
            parent / f"{stem.replace('.stems', '')}.key.json",
            *sorted(parent.glob("*.key.json")),
        ]
        for path in candidates:
            if path.is_file():
                try:
                    return json.loads(path.read_text()).get("key")
                except (json.JSONDecodeError, OSError):
                    continue
        return None

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

    def _clear_solo(self) -> None:
        self.engine.clear_solo()
        self._refresh_lanes()

    def _unmute_all(self) -> None:
        self.engine.unmute_all()
        self._refresh_lanes()

    def _refresh_lanes(self) -> None:
        for lane in self._lanes:
            lane.sync()

    # -- export ----------------------------------------------------------
    def _sync_export_controls(self) -> None:
        grand = self.staff_mode.currentData() == "grand"
        self.staff_hint.setText(
            "tick 1 (treble) / 2 (bass) per instrument" if grand else "tick 1 to include"
        )
        for lane in self._lanes:
            lane.set_grand_mode(grand)

    def _export(self) -> None:
        midi = self.engine.song.path
        if midi is None:
            return
        grand = self.staff_mode.currentData() == "grand"
        if grand:
            staves = [
                [lane.track.index for lane in self._lanes if lane.on_staff1()],
                [lane.track.index for lane in self._lanes if lane.on_staff2()],
            ]
        else:
            staves = [[lane.track.index for lane in self._lanes if lane.on_staff1()]]
        if not any(staves):
            QMessageBox.information(
                self, "Export", "Tick a staff checkbox on one or more instruments first."
            )
            return

        formats = []
        if self.fmt_musicxml.isChecked():
            formats.append("musicxml")
        if self.fmt_abc.isChecked():
            formats.append("abc")
        if not formats:
            QMessageBox.information(self, "Export", "Choose at least one format (MusicXML/ABC).")
            return

        mode = "grand" if grand else "single"
        midi_path = Path(midi)
        out_dir = midi_path.parent
        basename = f"{midi_path.stem}.{mode}"
        quantize_divisors = self.grid_combo.currentData()
        formats_tuple = tuple(formats)
        apply_key = self.apply_key_box.isChecked() and self._detected_key
        key = self._detected_key if apply_key else None

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
                    title=midi_path.stem,
                )
                self.export_done.emit(result)
            except Exception as exc:  # delivered to the GUI thread via the signal
                self.export_done.emit(exc)

        threading.Thread(target=worker, daemon=True).start()

    def _on_export_done(self, result: object) -> None:
        self.export_btn.setEnabled(True)
        self.export_btn.setText("Export selected →")
        if isinstance(result, Exception):
            QMessageBox.critical(self, "Export failed", str(result))
            return
        assert isinstance(result, dict)
        lines = []
        for key, path in result.items():
            label = "ABC (skipped)" if key == "abc_error" else key
            lines.append(f"{label}: {path}")
        QMessageBox.information(self, "Export complete", "\n".join(lines))

    def _refresh_playheads(self) -> None:
        for lane in self._lanes:
            lane.refresh_playhead()

    # -- periodic update -------------------------------------------------
    def _tick(self) -> None:
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
