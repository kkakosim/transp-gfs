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
