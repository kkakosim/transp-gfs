"""Allow ``python -m gfs2calmet`` to invoke the CLI."""

from gfs2calmet.cli import main


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(main())
