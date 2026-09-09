"""MPE expression engine: velocity, pressure, CC74 and vibrato from song artifacts.

Takes a (transcribed or score-derived) MIDI file plus the per-song analysis
artifacts (``meter.json``, ``key.json``, ``chords.json``, ``sections.json``)
and renders an expressive MPE performance — either to a ``.mpe.mid`` file
(``sound2midi-mpe``) or to a realtime virtual MIDI port for a DAW
(``sound2midi-mpe-play``).

Not to be confused with the unrelated ``sound2midi-mei`` notation exporter
(MEI = Music Encoding Initiative); MPE here is MIDI Polyphonic Expression.
"""
