"""Allow ``python -m tradebot`` to use the research-only operator CLI."""

from .app.cli import main


raise SystemExit(main())
