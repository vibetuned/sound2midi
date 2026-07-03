"""Song-structure strip for the player.

Reads the segments of a ``<song>.sections.json`` artifact (SongFormer, written by
``sound2midi --sections``) and draws them as one row of colored, labeled blocks
above the instrument lanes — same full-song-width time axis as the note strips.
Clicking sections toggles them in and out of the **selection**: playback loops
over the selected sections in song order (skipping the gaps), and exports are cut
to them. The loop itself is enforced by the window's tick timer, so the strip
stays a plain widget.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFontMetrics, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

from sound2midi.player.engine import PlayerEngine
from sound2midi.player.pianoroll import _LANE_BG, _PLAYHEAD

# One hue per SongForm-HX-8Class label; unknown labels get a hashed hue.
_LABEL_HUES = {
    "intro": 190,  # cyan
    "verse": 215,  # blue
    "pre-chorus": 45,  # amber
    "prechorus": 45,
    "chorus": 0,  # red
    "bridge": 280,  # purple
    "inst": 130,  # green
    "solo": 130,
    "outro": 250,  # indigo
}


def section_color(label: str, *, selected: bool = False) -> QColor:
    color = QColor()
    if label == "silence":
        color.setHsv(0, 0, 165 if selected else 140)
        return color
    hue = _LABEL_HUES.get(label, hash(label) % 360)
    color.setHsv(hue, 200 if selected else 150, 235 if selected else 205)
    return color


@dataclass
class Section:
    label: str
    start: float
    end: float
    display: str  # e.g. "Chorus 2" — repeated labels are numbered


def build_sections(segments: list[dict]) -> list[Section]:
    """Turn artifact segments into Sections with numbered display names."""
    sections: list[Section] = []
    for seg in segments:
        try:
            label = str(seg["label"])
            start, end = float(seg["start"]), float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            sections.append(Section(label, start, end, label.capitalize()))
    counts: dict[str, int] = {}
    for sec in sections:
        counts[sec.label] = counts.get(sec.label, 0) + 1
    seen: dict[str, int] = {}
    for sec in sections:
        if counts[sec.label] > 1:
            seen[sec.label] = seen.get(sec.label, 0) + 1
            sec.display = f"{sec.display} {seen[sec.label]}"
    return sections


class SectionStrip(QWidget):
    """Paints the sections across the full song width; click toggles selection."""

    def __init__(
        self,
        sections: list[Section],
        engine: PlayerEngine,
        on_seek: Callable[[], None] | None = None,
        on_selection: Callable[[list[int]], None] | None = None,
    ) -> None:
        super().__init__()
        self.sections = sections
        self.engine = engine
        self.selected: set[int] = set()
        self._on_seek = on_seek
        self._on_selection = on_selection
        self.setMinimumHeight(26)
        self.setMinimumWidth(160)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(
            "Click sections to select them: playback loops the selection (in song "
            "order) and exports are cut to it. Click a selected section to deselect. "
            "Ctrl+click an instrument's lane to pick instruments per section."
        )

    def _index_at(self, seconds: float) -> int | None:
        for i, sec in enumerate(self.sections):
            if sec.start <= seconds < sec.end:
                return i
        return None

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), _LANE_BG)
        duration = self.engine.duration or 1.0
        w, h = self.width(), self.height()

        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        metrics = QFontMetrics(font)

        for i, sec in enumerate(self.sections):
            x0 = sec.start / duration * w
            x1 = min(sec.end, duration) / duration * w
            if x1 - x0 < 1.0:
                continue
            rect = QRectF(x0, 1.0, x1 - x0 - 1.0, h - 2.0)
            selected = i in self.selected
            painter.setBrush(section_color(sec.label, selected=selected))
            painter.setPen(QPen(QColor(255, 255, 255), 2) if selected else Qt.PenStyle.NoPen)
            painter.drawRect(rect)
            if rect.width() >= 30:
                text = metrics.elidedText(
                    sec.display, Qt.TextElideMode.ElideRight, int(rect.width()) - 6
                )
                painter.setPen(QColor(20, 20, 22))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

        x = int(self.engine.position() / duration * w)
        painter.setPen(QPen(_PLAYHEAD, 1))
        painter.drawLine(x, 0, x, h)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        duration = self.engine.duration
        if duration <= 0 or not self.sections:
            return
        seconds = event.position().x() / max(1, self.width()) * duration
        index = self._index_at(seconds)
        if index is None:
            return
        if index in self.selected:
            self.selected.discard(index)
        else:
            self.selected.add(index)
            self.engine.seek(self.sections[index].start)
            if self._on_seek is not None:
                self._on_seek()
        if self._on_selection is not None:
            self._on_selection(sorted(self.selected))
        self.update()


class SectionLane(QFrame):
    """The sections header row: a control-column label beside the strip.

    Mirrors :class:`InstrumentLane`'s geometry so the strip's time axis lines up
    exactly with the note strips below it.
    """

    def __init__(
        self,
        sections: list[Section],
        engine: PlayerEngine,
        *,
        on_seek: Callable[[], None] | None = None,
        on_selection: Callable[[list[int]], None] | None = None,
    ) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.NoFrame)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 1, 4, 1)
        layout.setSpacing(4)

        control = QWidget()
        control.setFixedWidth(210)  # keep in sync with InstrumentLane's control column
        crow = QHBoxLayout(control)
        crow.setContentsMargins(0, 0, 0, 0)
        crow.setSpacing(3)
        title = QLabel("Sections")
        title.setStyleSheet("color: #888;")
        self.loop_label = QLabel("")
        self.loop_label.setStyleSheet("color: #6abf69;")
        crow.addWidget(title)
        crow.addWidget(self.loop_label, 1)

        def selection_changed(indices: list[int]) -> None:
            if not indices:
                text = ""
            elif len(indices) == 1:
                text = f"⟳ {sections[indices[0]].display}"
            else:
                text = f"⟳ {len(indices)} sections"
            self.loop_label.setText(text)
            if on_selection is not None:
                on_selection(indices)

        self.strip = SectionStrip(sections, engine, on_seek=on_seek, on_selection=selection_changed)

        layout.addWidget(control)
        layout.addWidget(self.strip, 1)
        self.setMinimumHeight(30)

    def refresh_playhead(self) -> None:
        self.strip.update()
