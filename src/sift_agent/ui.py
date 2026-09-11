"""Console entry for the Streamlit UI."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print("streamlit is not installed. Run: uv sync --extra ui")
        raise SystemExit(1)

    app = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app), *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
