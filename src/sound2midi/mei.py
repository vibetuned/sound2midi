"""MusicXML -> MEI conversion (Verovio 5.7) with a lead-in "measure 0".

Recreates the abc2mei pipeline tail for sound2midi's exported MusicXML:

    MusicXML --[music21: load + zero-measure transform]--> MusicXML
             --[Verovio + normalise]--> MEI

The lead-in measure holds a single quarter rest, is numbered 0, and carries the
part's opening clef / key / time signature so Verovio still lifts them into the
top ``<scoreDef>``. Verovio 5.7 emits clean MEI 5.1 directly; the only tidying
needed is for artifacts of the music21 round-trip (an inflated ``@ppq`` of
10080, ``metcon`` flags on the incomplete lead-in bar).

CLI: ``sound2midi-mei <file.musicxml> ...`` — a file living in a ``musicxml/``
folder (the export layout) lands in a sibling ``mei/`` folder by default.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import verovio
from lxml import etree  # ty: ignore[unresolved-import]  # compiled C extension
from music21 import clef, converter, key, meter, note, stream
from music21.musicxml.m21ToXml import GeneralObjectExporter
from music21.stream import Opus, Score

MEI_NS = "http://www.music-encoding.org/ns/mei"
DEFAULT_MEI_VERSION = "5.1"
# Clean pulses-per-quarter for the output (the music21 round-trip inflates it
# to 10080, which breaks MIDI consumers).
DEFAULT_PPQ = "128"

_NS = {"m": MEI_NS}

# The lead-in always contains exactly one quarter rest.
_REST_QL = 1.0
# Fallback bar length when a part declares no time signature.
_DEFAULT_BAR_QL = 4.0


# --------------------------------------------------------------------------
# music21 I/O
# --------------------------------------------------------------------------


def load_musicxml(musicxml: str) -> Score:
    """Parse a MusicXML document into a music21 score."""
    parsed = converter.parse(musicxml, format="musicxml")
    if isinstance(parsed, Opus):
        parsed = parsed.scores[0]
    if isinstance(parsed, Score):
        return parsed
    # A bare Part (or other stream) gets wrapped so callers always get a Score.
    wrapper = Score()
    wrapper.append(parsed)
    return wrapper


def dump_musicxml(score: Score) -> str:
    """Serialise a music21 score back to a MusicXML document."""
    return GeneralObjectExporter(score).parse().decode("utf-8")


# --------------------------------------------------------------------------
# Transform: prepend a "measure 0" holding a single quarter rest
# --------------------------------------------------------------------------


def add_zero_measure(score: Score) -> Score:
    """Prepend a quarter-rest lead-in measure (n=0) to every part in ``score``.

    The lead-in is metrically incomplete (one quarter rest, regardless of the
    time signature). The part's opening attributes move onto it so the MEI
    scoreDef stays complete; a treble clef is supplied when the source has none.
    """
    for part in score.parts:
        measures = part.getElementsByClass(stream.Measure)
        if not measures:
            continue
        first = measures[0]

        time_sig = first.getElementsByClass(meter.TimeSignature).first()
        bar_ql = time_sig.barDuration.quarterLength if time_sig else _DEFAULT_BAR_QL

        lead_in = stream.Measure(number=0)

        existing_clef = first.getElementsByClass(clef.Clef).first()
        if existing_clef is not None:
            first.remove(existing_clef)
            lead_in.insert(0, existing_clef)
        else:
            lead_in.insert(0, clef.TrebleClef())
        for attr_cls in (key.KeySignature, meter.TimeSignature):
            obj = first.getElementsByClass(attr_cls).first()
            if obj is not None:
                first.remove(obj)
                lead_in.insert(0, obj)

        lead_in.insert(0, note.Rest(quarterLength=_REST_QL))
        # Mark the bar as intentionally short so music21 does not pad it out.
        lead_in.paddingRight = max(0.0, bar_ql - _REST_QL)

        part.insertAndShift(0.0, lead_in)
    return score


# --------------------------------------------------------------------------
# MusicXML -> MEI via Verovio, plus normalisation
# --------------------------------------------------------------------------


def musicxml_to_mei(musicxml: str, *, mei_version: str = DEFAULT_MEI_VERSION) -> str:
    """Render MusicXML to MEI and normalise it to ``mei_version``."""
    toolkit = verovio.toolkit()
    toolkit.setInputFrom("musicxml")
    if not toolkit.loadData(musicxml):
        raise ValueError("Verovio failed to load the MusicXML data")
    mei = toolkit.getMEI()
    return _normalise(mei, mei_version)


def convert_musicxml_to_mei(
    musicxml: str,
    *,
    zero_measure: bool = True,
    mei_version: str = DEFAULT_MEI_VERSION,
) -> str:
    """Convert a MusicXML document to MEI, optionally adding the lead-in measure."""
    if zero_measure:
        score = add_zero_measure(load_musicxml(musicxml))
        musicxml = dump_musicxml(score)
    return musicxml_to_mei(musicxml, mei_version=mei_version)


def _normalise(mei: str, mei_version: str) -> str:
    root = etree.fromstring(mei.encode("utf-8"))
    _normalise_timing(root)
    _strip_metcon(root)
    _clean_meihead(root)
    root.set("meiversion", mei_version)
    etree.indent(root, space="   ")  # re-indent (Verovio uses 3 spaces)
    body = etree.tostring(root, encoding="unicode")
    return _prolog(mei_version) + body + "\n"


def _normalise_timing(root: etree._Element) -> None:
    """Rescale the music21-inflated timing resolution to a clean value.

    The music21 round-trip makes Verovio inflate ``@ppq`` to 10080 (music21's
    default ``divisionsPerQuarter``) and stamp a matching ``dur.ppq`` on every
    event. Both are rescaled to :data:`DEFAULT_PPQ` rather than dropped.
    """
    source_ppq = next((int(el.get("ppq", "")) for el in root.iter() if el.get("ppq")), None)
    if not source_ppq:
        return
    target_ppq = int(DEFAULT_PPQ)
    for element in root.iter():
        if "ppq" in element.attrib:
            element.set("ppq", DEFAULT_PPQ)
        dur_ppq = element.get("dur.ppq")
        if dur_ppq is not None:
            element.set("dur.ppq", str(round(int(dur_ppq) * target_ppq / source_ppq)))

    # The inserted lead-in (n="0") is a free measure; omit its gestural timing.
    for measure in root.findall('.//m:measure[@n="0"]', _NS):
        for element in measure.iter():
            element.attrib.pop("dur.ppq", None)


def _strip_metcon(root: etree._Element) -> None:
    """Remove Verovio's ``metcon="false"`` flag from measures.

    Verovio stamps ``metcon`` (metrically conforming = false) on any incomplete
    bar — the lead-in ``n="0"`` in particular. The consuming app does not want
    it, so it is dropped everywhere.
    """
    for measure in root.findall(".//m:measure", _NS):
        measure.attrib.pop("metcon", None)


def _clean_meihead(root: etree._Element) -> None:
    """Remove the placeholder ``Music21`` composer music21 sometimes inserts.

    Real composers (sound2midi stamps itself on exports) survive; only the
    literal ``Music21`` placeholder goes, leaving an empty ``<respStmt/>``.
    """
    for pers in root.findall(".//m:respStmt/m:persName", _NS):
        if (pers.text or "").strip() == "Music21":
            parent = pers.getparent()
            if parent is not None:
                parent.remove(pers)
                if len(parent) == 0:
                    parent.text = None  # self-close the now-empty <respStmt/>


def _prolog(mei_version: str) -> str:
    schema = f"https://music-encoding.org/schema/{mei_version}/mei-all.rng"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<?xml-model href="{schema}" type="application/xml" '
        'schematypens="http://relaxng.org/ns/structure/1.0"?>\n'
        f'<?xml-model href="{schema}" type="application/xml" '
        'schematypens="http://purl.oclc.org/dsdl/schematron"?>\n'
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _output_path(inp: Path, output: Path | None, many: bool) -> Path:
    if output is None:
        # Exports live in <song>/musicxml/; their MEI goes to a sibling mei/.
        if inp.parent.name == "musicxml":
            out_dir = inp.parent.parent / "mei"
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir / inp.with_suffix(".mei").name
        return inp.with_suffix(".mei")
    if many or output.is_dir():
        output.mkdir(parents=True, exist_ok=True)
        return output / inp.with_suffix(".mei").name
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sound2midi-mei",
        description="Convert exported MusicXML files to MEI (Verovio 5.7), "
        "prepending a quarter-rest lead-in measure 0.",
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="MusicXML input file(s)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output file (single input) or directory (multiple inputs); by default "
        "a file in a musicxml/ folder lands in a sibling mei/ folder, anything "
        "else alongside the input with a .mei suffix",
    )
    parser.add_argument(
        "--mei-version",
        default=DEFAULT_MEI_VERSION,
        help="target MEI version (default: %(default)s)",
    )
    parser.add_argument(
        "--no-zero-measure",
        action="store_true",
        help="do not prepend the leading zero measure",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    failures = 0
    for inp in args.inputs:
        try:
            mei = convert_musicxml_to_mei(
                inp.read_text(encoding="utf-8"),
                zero_measure=not args.no_zero_measure,
                mei_version=args.mei_version,
            )
        except Exception as err:  # report and continue with remaining files
            print(f"error: {inp}: {err}", file=sys.stderr)
            failures += 1
            continue
        out = _output_path(inp, args.output, many=len(args.inputs) > 1)
        out.write_text(mei, encoding="utf-8")
        print(f"{inp} -> {out}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
