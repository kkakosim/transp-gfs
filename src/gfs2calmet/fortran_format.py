"""FORTRAN-equivalent fixed-width formatters.

The CALMET 3D.DAT v2.1 spec is defined in terms of FORTRAN FORMAT
statements (see Table 7-33 of the CALPUFF v6 User Instructions). We
emulate the relevant edit descriptors here so the writer can produce
byte-identical output to the reference CALWRF/CALMM5 implementations.

Edit descriptors covered:
    Iw       — integer, right-justified in field of width w
    Iw.m     — integer, right-justified, zero-padded to at least m digits
    Fw.d     — fixed-point real with d decimals in field of width w
    Aw       — character, left-justified, padded with spaces

On overflow we emit asterisks ('*' * w), matching FORTRAN behavior.
"""

from __future__ import annotations


def fmt_i(value: int, width: int, min_digits: int | None = None) -> str:
    """FORTRAN Iw or Iw.m.

    If ``min_digits`` is given the digit portion is zero-padded so the
    result has at least that many digits (excluding the sign). This is
    the FORTRAN ``Iw.m`` form, used by 3D.DAT to encode date components
    as ``i2.2`` (e.g. ``01`` for January) inside a concatenated YYYYMMDDHH
    string.
    """
    if not isinstance(value, int):
        raise TypeError(f"fmt_i requires int, got {type(value).__name__}")
    if min_digits is not None and min_digits > width:
        raise ValueError("min_digits cannot exceed width")

    sign = "-" if value < 0 else ""
    digits = str(abs(value))
    if min_digits is not None and len(digits) < min_digits:
        digits = digits.zfill(min_digits)

    s = sign + digits
    if len(s) > width:
        return "*" * width
    return s.rjust(width)


def fmt_f(value: float, width: int, decimals: int) -> str:
    """FORTRAN Fw.d."""
    s = f"{value:{width}.{decimals}f}"
    if len(s) > width:
        return "*" * width
    return s


def fmt_a(value: str, width: int) -> str:
    """FORTRAN Aw.

    On input that is shorter than ``width`` FORTRAN pads with trailing
    spaces. On input that is longer FORTRAN truncates to the leftmost
    ``width`` characters.
    """
    if len(value) >= width:
        return value[:width]
    return value + " " * (width - len(value))


def fmt_x(count: int) -> str:
    """FORTRAN nX edit descriptor — emit ``count`` blank characters."""
    return " " * count


# ---------------------------------------------------------------------------
# ASCII sanitization for text fields written to 3D.DAT
# ---------------------------------------------------------------------------


# Common typography that users paste into YAML configs but that CALMET
# (an ASCII-only FORTRAN reader) cannot parse. Replace before writing.
_UNICODE_TO_ASCII: dict[str, str] = {
    "—": "-",      # em dash
    "–": "-",      # en dash
    "‘": "'",      # left single quote
    "’": "'",      # right single quote
    "“": '"',      # left double quote
    "”": '"',      # right double quote
    "…": "...",    # ellipsis
    "°": " deg",   # degree sign
    "×": "x",      # multiplication sign
    "µ": "u",      # micro sign
    "→": "->",     # right arrow
    "←": "<-",     # left arrow
    " ": " ",      # non-breaking space
}


def to_ascii(value: str) -> str:
    """Best-effort transliteration of ``value`` to pure ASCII.

    Maps a curated set of common typography (em/en dashes, smart quotes,
    ellipsis, degree sign, etc.) to ASCII equivalents. Any character
    that remains outside the ASCII range is replaced with ``"?"`` so
    the writer never produces a file CALMET cannot read.

    Pure-ASCII input is returned unchanged (no copy).
    """
    if value.isascii():
        return value
    for src, dst in _UNICODE_TO_ASCII.items():
        if src in value:
            value = value.replace(src, dst)
    # Anything still non-ASCII becomes '?'.
    return value.encode("ascii", "replace").decode("ascii")
