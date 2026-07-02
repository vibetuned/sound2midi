"""Per-instrument piano-roll lanes for the player.

Each instrument gets an :class:`InstrumentLane`: a control box (name + Solo/Mute) next
to a :class:`NoteStrip` that draws that track's notes over time (gaps between notes are
the rests). A red playhead sweeps across during playback; clicking a strip seeks.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen, QPixmap, QResizeEvent
from PySide6.QtWidgets import QCheckBox, QFrame, QHBoxLayout, QLabel, QPushButton, QWidget

from sound2midi.player.engine import PlayerEngine, TrackInfo

_SOLO_STYLE = "QPushButton:checked { background: #2e7d32; color: white; }"
_MUTE_STYLE = "QPushButton:checked { background: #c62828; color: white; }"

_GRID = QColor(70, 70, 76)
_PLAYHEAD = QColor(255, 80, 80)
_LANE_BG = QColor(43, 43, 48)
_LANE_BG_MUTED = QColor(30, 30, 33)


def track_color(index: int, n_tracks: int) -> QColor:
    """A distinct, evenly-spaced hue per track."""
    color = QColor()
    color.setHsv(int(360 * (index / max(1, n_tracks))) % 360, 170, 235)
    return color


class NoteStrip(QWidget):
    """Paints one track's notes across the full song width, with a playhead."""

    def __init__(
        self,
        track: TrackInfo,
        engine: PlayerEngine,
        n_tracks: int,
        on_seek: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.track = track
        self.engine = engine
        self.color = track_color(track.index, n_tracks)
        self._on_seek = on_seek
        self._cache: QPixmap | None = None
        self.setMinimumHeight(26)
        self.setMinimumWidth(160)

        pitches = [p for _, _, p in track.notes]
        self._pmin = min(pitches) if pitches else 60
        self._pmax = max(pitches) if pitches else 72
        if self._pmax - self._pmin < 8:  # give very narrow ranges some headroom
            self._pmin -= 4
            self._pmax += 4

    def invalidate(self) -> None:
        self._cache = None
        self.update()

    def _render_cache(self) -> None:
        w, h = max(1, self.width()), max(1, self.height())
        pm = QPixmap(w, h)
        audible = self.engine.is_audible(self.track.index)
        pm.fill(_LANE_BG if audible else _LANE_BG_MUTED)

        painter = QPainter(pm)
        duration = self.engine.duration or 1.0

        # faint 10-second gridlines for time reference
        painter.setPen(QPen(_GRID, 1))
        step = 10.0
        t = step
        while t < duration:
            x = int(t / duration * w)
            painter.drawLine(x, 0, x, h)
            t += step

        note_color = QColor(self.color)
        if not audible:
            note_color.setAlpha(55)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(note_color)

        pad = 2.0
        usable = max(1.0, h - 2 * pad)
        span = max(1, self._pmax - self._pmin)
        row_h = max(2.0, usable / span)
        for start, end, pitch in self.track.notes:
            x0 = start / duration * w
            x1 = end / duration * w
            y = pad + (1.0 - (pitch - self._pmin) / span) * usable
            painter.drawRect(QRectF(x0, y - row_h / 2, max(1.5, x1 - x0), row_h))
        painter.end()
        self._cache = pm

    def paintEvent(self, event: QPaintEvent) -> None:
        if self._cache is None or self._cache.size() != self.size():
            self._render_cache()
        painter = QPainter(self)
        if self._cache is not None:
            painter.drawPixmap(0, 0, self._cache)
        duration = self.engine.duration or 1.0
        x = int(self.engine.position() / duration * self.width())
        painter.setPen(QPen(_PLAYHEAD, 1))
        painter.drawLine(x, 0, x, self.height())
        painter.end()

    def resizeEvent(self, event: QResizeEvent) -> None:
        self._cache = None
        super().resizeEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        duration = self.engine.duration
        if duration > 0:
            self.engine.seek(event.position().x() / max(1, self.width()) * duration)
            if self._on_seek is not None:
                self._on_seek()
            self.update()


class InstrumentLane(QFrame):
    """An instrument's controls (name + Solo + Mute) beside its note strip."""

    def __init__(
        self,
        track: TrackInfo,
        engine: PlayerEngine,
        n_tracks: int,
        *,
        on_seek: Callable[[], None] | None = None,
        on_toggle: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.track = track
        self.engine = engine
        self._on_toggle = on_toggle
        self.setFrameShape(QFrame.Shape.NoFrame)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 1, 4, 1)
        layout.setSpacing(4)

        control = QWidget()
        control.setFixedWidth(180)
        crow = QHBoxLayout(control)
        crow.setContentsMargins(0, 0, 0, 0)
        crow.setSpacing(3)

        self.staff1_box = QCheckBox("1")
        self.staff1_box.setToolTip("Add to staff 1 (top / treble) for export")
        self.staff2_box = QCheckBox("2")
        self.staff2_box.setToolTip("Add to staff 2 (bottom / bass) for export")
        self.staff2_box.setVisible(False)  # shown only in grand-staff mode

        swatch = QLabel()
        swatch.setFixedWidth(8)
        swatch.setStyleSheet(f"background: {track_color(track.index, n_tracks).name()};")
        name = QLabel(track.name)
        name.setMinimumWidth(78)
        name.setToolTip(f"{track.name} — {track.note_count} notes")

        self.solo_btn = QPushButton("S")
        self.solo_btn.setCheckable(True)
        self.solo_btn.setFixedWidth(26)
        self.solo_btn.setToolTip("Solo")
        self.solo_btn.setStyleSheet(_SOLO_STYLE)
        self.solo_btn.toggled.connect(self._on_solo)

        self.mute_btn = QPushButton("M")
        self.mute_btn.setCheckable(True)
        self.mute_btn.setFixedWidth(26)
        self.mute_btn.setToolTip("Mute")
        self.mute_btn.setStyleSheet(_MUTE_STYLE)
        self.mute_btn.toggled.connect(self._on_mute)

        crow.addWidget(self.staff1_box)
        crow.addWidget(self.staff2_box)
        crow.addWidget(swatch)
        crow.addWidget(name, 1)
        crow.addWidget(self.solo_btn)
        crow.addWidget(self.mute_btn)

        self.strip = NoteStrip(track, engine, n_tracks, on_seek=on_seek)

        layout.addWidget(control)
        layout.addWidget(self.strip, 1)
        self.setMinimumHeight(30)

    def _on_solo(self, checked: bool) -> None:
        self.engine.set_solo(self.track.index, checked)
        if self._on_toggle is not None:
            self._on_toggle()

    def _on_mute(self, checked: bool) -> None:
        self.engine.set_muted(self.track.index, checked)
        if self._on_toggle is not None:
            self._on_toggle()

    def sync(self) -> None:
        """Reflect engine state in the buttons and re-shade the strip."""
        self.solo_btn.blockSignals(True)
        self.mute_btn.blockSignals(True)
        self.solo_btn.setChecked(self.engine.is_solo(self.track.index))
        self.mute_btn.setChecked(self.engine.is_muted(self.track.index))
        self.solo_btn.blockSignals(False)
        self.mute_btn.blockSignals(False)
        self.strip.invalidate()

    def refresh_playhead(self) -> None:
        self.strip.update()

    def set_grand_mode(self, grand: bool) -> None:
        """Show the staff-2 checkbox only when exporting a grand (2-staff) score."""
        self.staff2_box.setVisible(grand)
        if not grand:
            self.staff2_box.setChecked(False)

    def on_staff1(self) -> bool:
        return self.staff1_box.isChecked()

    def on_staff2(self) -> bool:
        return self.staff2_box.isChecked()
