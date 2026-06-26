"""Tests for the FORTRAN-equivalent fixed-width formatters."""

import pytest

from gfs2calmet.fortran_format import fmt_a, fmt_f, fmt_i, fmt_x, to_ascii


class TestFmtI:
    def test_right_justifies_positive(self) -> None:
        assert fmt_i(7, 4) == "   7"

    def test_right_justifies_negative(self) -> None:
        assert fmt_i(-3, 4) == "  -3"

    def test_zero(self) -> None:
        assert fmt_i(0, 3) == "  0"

    def test_exact_width(self) -> None:
        assert fmt_i(1234, 4) == "1234"

    def test_overflow_emits_asterisks(self) -> None:
        # FORTRAN behavior: value too wide for field becomes asterisks.
        assert fmt_i(12345, 4) == "****"

    def test_min_digits_zero_pads_month(self) -> None:
        # Equivalent to FORTRAN I2.2 — encodes "01" for January
        # inside the YYYYMMDDHH concatenation.
        assert fmt_i(1, 2, min_digits=2) == "01"
        assert fmt_i(12, 2, min_digits=2) == "12"

    def test_min_digits_zero_pads_year(self) -> None:
        assert fmt_i(2026, 4, min_digits=4) == "2026"
        assert fmt_i(99, 4, min_digits=4) == "0099"

    def test_min_digits_cannot_exceed_width(self) -> None:
        with pytest.raises(ValueError):
            fmt_i(1, 2, min_digits=3)

    def test_negative_with_min_digits(self) -> None:
        # Sign is preserved; digits are zero-padded.
        assert fmt_i(-5, 4, min_digits=2) == " -05"

    def test_requires_int(self) -> None:
        with pytest.raises(TypeError):
            fmt_i(1.0, 4)  # type: ignore[arg-type]


class TestFmtF:
    def test_basic(self) -> None:
        assert fmt_f(1012.3, 7, 1) == " 1012.3"

    def test_negative(self) -> None:
        assert fmt_f(-4.0, 6, 3) == "-4.000"

    def test_small_positive(self) -> None:
        assert fmt_f(0.03, 5, 2) == " 0.03"

    def test_overflow_emits_asterisks(self) -> None:
        assert fmt_f(1e9, 5, 1) == "*****"

    def test_zero(self) -> None:
        assert fmt_f(0.0, 6, 3) == " 0.000"


class TestFmtA:
    def test_left_justifies_with_space_pad(self) -> None:
        assert fmt_a("LCC", 4) == "LCC "

    def test_truncates_overlong(self) -> None:
        assert fmt_a("VERYLONGNAME", 4) == "VERY"

    def test_exact_width(self) -> None:
        assert fmt_a("ABCD", 4) == "ABCD"

    def test_empty_string(self) -> None:
        assert fmt_a("", 5) == "     "


class TestFmtX:
    def test_emits_blanks(self) -> None:
        assert fmt_x(3) == "   "

    def test_zero_blanks(self) -> None:
        assert fmt_x(0) == ""


class TestToAscii:
    def test_pure_ascii_passes_through_unchanged(self) -> None:
        s = "Qatar UTM 39N driver grid"
        assert to_ascii(s) is s  # same object, no copy

    def test_em_dash_becomes_hyphen(self) -> None:
        assert to_ascii("3D.DAT v2.1 — feed CALMET") == "3D.DAT v2.1 - feed CALMET"

    def test_en_dash_becomes_hyphen(self) -> None:
        assert to_ascii("range 1–5") == "range 1-5"

    def test_smart_quotes_become_straight_quotes(self) -> None:
        assert to_ascii("‘single’") == "'single'"
        assert to_ascii("“double”") == '"double"'

    def test_degree_sign_becomes_deg(self) -> None:
        assert to_ascii("25° N") == "25 deg N"

    def test_ellipsis_becomes_three_dots(self) -> None:
        assert to_ascii("more…") == "more..."

    def test_unmapped_unicode_becomes_question_mark(self) -> None:
        # CJK character not in our typography table.
        assert to_ascii("中") == "?"

    def test_mixed_typography_handled(self) -> None:
        s = "Qatar — 25°N ‘test’ …"
        assert to_ascii(s) == "Qatar - 25 degN 'test' ..."

    def test_result_is_always_ascii_encodable(self) -> None:
        # Pathological input still produces an ASCII-safe string.
        s = "—…中µ×°"
        out = to_ascii(s)
        out.encode("ascii")  # must not raise
