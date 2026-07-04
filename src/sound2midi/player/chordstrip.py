"""Chord strip for the player.

Reads the segments of a ``<song>.chords.json`` artifact (lv-chordia, written by
``sound2midi --chords``) and draws them as one row of colored blocks under the
section strip — same full-song-width time axis as the note strips. Blocks are
colored by the chord's root pitch class (minor-family chords darker); no-chord
spans stay lane-colored. Clicking seeks; hovering shows the chord.

The progression is also *playable*: the window realizes it as a synthesized
piano track in the engine (bass note + chord tones), and the lane carries the
same Solo/Mute buttons as an instrument lane — muted by default, so hit **S**
to audition the chords alone or untick **M** to comp along with the band.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFontMetrics, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QCheckBox, QComboBox, QFrame, QHBoxLayout, QPushButton, QWidget

from sound2midi.player import chordlabel
from sound2midi.player.engine import PlayerEngine
from sound2midi.player.pianoroll import _LANE_BG, _MUTE_STYLE, _PLAYHEAD, _SOLO_STYLE


def chord_color(label: str) -> QColor | None:
    """Root pitch class -> hue (C=red wheel start); minor family darker; N -> None."""
    parsed = chordlabel.parse(label)
    if parsed is None:
        return None
    hue = (chordlabel.ROOT_PC[parsed.root] * 30) % 360
    kind = chordlabel.base_kind(parsed.kind)
    minorish = kind.startswith("min") or kind.startswith("dim") or kind == "hdim7"
    color = QColor()
    color.setHsv(hue, 160, 165 if minorish else 210)
    return color


@dataclass
class ChordSeg:
    label: str
    start: float
    end: float
    display: str


def build_chords(segments: list[dict]) -> list[ChordSeg]:
    chords: list[ChordSeg] = []
    for seg in segments:
        try:
            label = str(seg["label"])
            start, end = float(seg["start"]), float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            chords.append(ChordSeg(label, start, end, chordlabel.display(label)))
    return chords


class ChordStrip(QWidget):
    """Paints the chord blocks across the full song width; click = seek."""

    def __init__(
        self,
        chords: list[ChordSeg],
        engine: PlayerEngine,
        on_seek: Callable[[], None] | None = None,
        track_index: int | None = None,
        sections: list[tuple[float, float]] | None = None,
    ) -> None:
        super().__init__()
        self.chords = chords
        self.engine = engine
        self.track_index = track_index  # synthesized chord track, if one was added
        self.sections = sections or []  # (start, end) per song section, in order
        self.cells: set[int] = set()  # section indices the chord voice is picked for
        self._on_seek = on_seek
        self.setMinimumHeight(22)
        self.setMinimumWidth(160)
        self.setMouseTracking(True)  # hover tooltip shows the chord under the cursor
        if self.sections:
            self.setToolTip(
                "Click: seek · Ctrl+click: toggle the chord voice for that section in the export"
            )

    def _chord_at(self, seconds: float) -> ChordSeg | None:
        for seg in self.chords:
            if seg.start <= seconds < seg.end:
                return seg
        return None

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), _LANE_BG)
        duration = self.engine.duration or 1.0
        w, h = self.width(), self.height()

        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)
        metrics = QFontMetrics(font)

        audible = self.track_index is None or self.engine.is_audible(self.track_index)
        for seg in self.chords:
            color = chord_color(seg.label)
            if color is None:
                continue  # no-chord span: leave the lane background
            if not audible:
                color.setAlpha(80)
            x0 = seg.start / duration * w
            x1 = min(seg.end, duration) / duration * w
            if x1 - x0 < 1.0:
                continue
            rect = QRectF(x0, 1.0, x1 - x0 - 1.0, h - 2.0)
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(rect)
            if rect.width() >= metrics.horizontalAdvance(seg.display) + 4:
                painter.setPen(QColor(20, 20, 22) if audible else QColor(120, 120, 126))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, seg.display)

        # highlight the sections the chord voice is picked for (export cells)
        for i in self.cells:
            start, end = self.sections[i]
            x0 = start / duration * w
            x1 = min(end, duration) / duration * w
            painter.fillRect(QRectF(x0, 0, x1 - x0, h), QColor(255, 255, 255, 45))

        x = int(self.engine.position() / duration * w)
        painter.setPen(QPen(_PLAYHEAD, 1))
        painter.drawLine(x, 0, x, h)
        painter.end()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        duration = self.engine.duration
        if duration <= 0:
            return
        seg = self._chord_at(event.position().x() / max(1, self.width()) * duration)
        self.setToolTip(f"{seg.display or 'N.C.'}  ({seg.start:.1f}-{seg.end:.1f}s)" if seg else "")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        duration = self.engine.duration
        if duration <= 0:
            return
        seconds = event.position().x() / max(1, self.width()) * duration
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and self.sections:
            for i, (start, end) in enumerate(self.sections):
                if start <= seconds < end:
                    self.cells.symmetric_difference_update({i})
                    self.update()
                    break
            return
        self.engine.seek(seconds)
        if self._on_seek is not None:
            self._on_seek()
        self.update()


class ChordLane(QFrame):
    """The chords header row: a control-column label beside the strip.

    Mirrors :class:`InstrumentLane`'s geometry so the strip's time axis lines up
    exactly with the note strips below it.
    """

    def __init__(
        self,
        chords: list[ChordSeg],
        engine: PlayerEngine,
        *,
        track_index: int | None = None,
        sections: list[tuple[float, float]] | None = None,
        on_seek: Callable[[], None] | None = None,
        on_toggle: Callable[[], None] | None = None,
        on_style: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.track_index = track_index
        self._on_toggle = on_toggle
        self.setFrameShape(QFrame.Shape.NoFrame)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 1, 4, 1)
        layout.setSpacing(4)

        control = QWidget()
        control.setFixedWidth(210)  # keep in sync with InstrumentLane's control column
        crow = QHBoxLayout(control)
        crow.setContentsMargins(0, 0, 0, 0)
        crow.setSpacing(3)

        self.staff1_box = QCheckBox("1")
        self.staff1_box.setToolTip("Add the realized chords to staff 1 (top / treble) for export")
        self.staff2_box = QCheckBox("2")
        self.staff2_box.setToolTip("Add the realized chords to staff 2 (bottom / bass) for export")
        self.staff2_box.setVisible(False)  # shown only in grand-staff mode
        crow.addWidget(self.staff1_box)
        crow.addWidget(self.staff2_box)

        self.style_combo = QComboBox()
        self.style_combo.addItem("Chords: block", "block")
        self.style_combo.addItem("Chords: smooth", "smooth")
        self.style_combo.addItem("Chords: arpeggio", "arpeggio")
        self.style_combo.addItem("Chords: bass", "bass")
        self.style_combo.setToolTip(
            "How the progression is realized (playback and export): block chords, "
            "voice-led inversions (smooth), one chord tone per beat (arpeggio), or "
            "the bass line only"
        )
        if on_style is not None:
            self.style_combo.currentIndexChanged.connect(
                lambda _i: on_style(self.style_combo.currentData())
            )
        crow.addWidget(self.style_combo, 1)

        self.solo_btn = QPushButton("S")
        self.solo_btn.setCheckable(True)
        self.solo_btn.setFixedWidth(26)
        self.solo_btn.setToolTip("Solo the synthesized chord piano")
        self.solo_btn.setStyleSheet(_SOLO_STYLE)
        self.solo_btn.toggled.connect(self._on_solo)
        self.mute_btn = QPushButton("M")
        self.mute_btn.setCheckable(True)
        self.mute_btn.setFixedWidth(26)
        self.mute_btn.setToolTip("Mute the synthesized chord piano (muted by default)")
        self.mute_btn.setStyleSheet(_MUTE_STYLE)
        self.mute_btn.toggled.connect(self._on_mute)
        if track_index is None:
            self.solo_btn.setVisible(False)
            self.mute_btn.setVisible(False)
        crow.addWidget(self.solo_btn)
        crow.addWidget(self.mute_btn)

        self.strip = ChordStrip(
            chords, engine, on_seek=on_seek, track_index=track_index, sections=sections
        )

        layout.addWidget(control)
        layout.addWidget(self.strip, 1)
        self.setMinimumHeight(26)
        self.sync()

    def _on_solo(self, checked: bool) -> None:
        if self.track_index is not None:
            self.engine.set_solo(self.track_index, checked)
            if self._on_toggle is not None:
                self._on_toggle()

    def _on_mute(self, checked: bool) -> None:
        if self.track_index is not None:
            self.engine.set_muted(self.track_index, checked)
            if self._on_toggle is not None:
                self._on_toggle()

    def sync(self) -> None:
        """Reflect engine state in the buttons and re-shade the strip."""
        if self.track_index is not None:
            self.solo_btn.blockSignals(True)
            self.mute_btn.blockSignals(True)
            self.solo_btn.setChecked(self.engine.is_solo(self.track_index))
            self.mute_btn.setChecked(self.engine.is_muted(self.track_index))
            self.solo_btn.blockSignals(False)
            self.mute_btn.blockSignals(False)
        self.strip.update()

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

    def chord_style(self) -> str:
        """The selected realization style ('block', 'smooth', 'arpeggio', 'bass')."""
        return str(self.style_combo.currentData())

    def cells(self) -> set[int]:
        """Section indices the chord voice is picked for (Ctrl+click on the strip)."""
        return set(self.strip.cells)
