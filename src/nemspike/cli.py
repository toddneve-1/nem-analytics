"""Command-line entry point. Sub-commands are added as each pipeline stage is built."""

from __future__ import annotations

import typer

app = typer.Typer(
    help="NEM VIC1 price spike forecasting and battery evaluation.", no_args_is_help=True
)


@app.command()
def version() -> None:
    """Print the package version."""
    from nemspike import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
