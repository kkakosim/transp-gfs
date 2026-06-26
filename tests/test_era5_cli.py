"""Tests for the ERA5 CLI helpers.

We do not exercise the full pipeline here (that requires cdsapi + a
network round-trip).  We test the small pure-Python helpers whose
behavior is observable.
"""

from __future__ import annotations

from datetime import datetime

from gfs2calmet.era5_cli import _resolve_output_path
from gfs2calmet.era5_reader import _split_into_monthly_chunks


class TestSplitIntoMonthlyChunks:
    def test_single_day_is_one_chunk(self) -> None:
        chunks = _split_into_monthly_chunks(
            datetime(2022, 7, 1, 0), datetime(2022, 7, 1, 23),
        )
        assert chunks == [(datetime(2022, 7, 1, 0), datetime(2022, 7, 1, 23))]

    def test_range_within_one_month_is_one_chunk(self) -> None:
        chunks = _split_into_monthly_chunks(
            datetime(2022, 7, 1, 0), datetime(2022, 7, 15, 23),
        )
        assert len(chunks) == 1

    def test_two_month_span_yields_two_chunks(self) -> None:
        chunks = _split_into_monthly_chunks(
            datetime(2022, 7, 15, 0), datetime(2022, 8, 5, 12),
        )
        assert chunks == [
            (datetime(2022, 7, 15, 0),  datetime(2022, 7, 31, 23)),
            (datetime(2022, 8, 1, 0),   datetime(2022, 8, 5, 12)),
        ]

    def test_year_boundary(self) -> None:
        chunks = _split_into_monthly_chunks(
            datetime(2022, 12, 30, 0), datetime(2023, 1, 2, 0),
        )
        assert chunks == [
            (datetime(2022, 12, 30, 0), datetime(2022, 12, 31, 23)),
            (datetime(2023, 1, 1, 0),   datetime(2023, 1, 2, 0)),
        ]

    def test_three_month_span(self) -> None:
        chunks = _split_into_monthly_chunks(
            datetime(2022, 6, 15, 0), datetime(2022, 8, 10, 0),
        )
        assert [c[0].month for c in chunks] == [6, 7, 8]
        assert chunks[0][1] == datetime(2022, 6, 30, 23)
        assert chunks[1][1] == datetime(2022, 7, 31, 23)
        assert chunks[2][1] == datetime(2022, 8, 10, 0)


class TestResolveOutputPath:
    def test_injects_start_and_end_when_missing(self) -> None:
        # User wrote no date in the filename → we inject both.
        out = _resolve_output_path(
            "./out/ERA5.3D.DAT",
            datetime(2022, 7, 1, 0), datetime(2022, 7, 4, 3),
        )
        assert "2022070100" in out
        assert "2022070403" in out
        assert out.endswith(".3D.DAT")

    def test_preserves_filename_when_start_stamp_present(self) -> None:
        # User provided an exact filename with the start stamp → leave alone.
        out = _resolve_output_path(
            "./out/ERA5_2022070100.3D.DAT",
            datetime(2022, 7, 1, 0), datetime(2022, 7, 4, 3),
        )
        assert out.endswith("ERA5_2022070100.3D.DAT")

    def test_strips_stale_date_stamp(self) -> None:
        # User's filename has the WRONG date → strip it and re-stamp so
        # CALMET.INP can never accidentally point at a stale file.
        out = _resolve_output_path(
            "./out/ERA5_2024011500.3D.DAT",
            datetime(2022, 7, 1, 0), datetime(2022, 7, 4, 3),
        )
        assert "2024011500" not in out
        assert "2022070100" in out
        assert "2022070403" in out

    def test_strips_stale_start_end_pair(self) -> None:
        # Filename already had a {start}_{end} pair (from a previous run)
        # → both get stripped and replaced.
        out = _resolve_output_path(
            "./out/ERA5_2021070100_2021070403.3D.DAT",
            datetime(2022, 7, 1, 0), datetime(2022, 7, 4, 3),
        )
        assert "2021" not in out
        assert "2022070100_2022070403" in out
