"""sift CLI — run the auto-EDA agent or ask it questions about a file."""

from __future__ import annotations

import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from .analysis import profile_dataframe
from .config import DEFAULT_MODELS, PROVIDERS, REQUIRED_ENV, get_llm
from .graph import run_pipeline
from .loader import LoadError, load_table
from .paths import unique_dir
from .qa import answer_question

app = typer.Typer(
    add_completion=False, help="sift — auto-EDA agent powered by LangGraph."
)
console = Console()


def _resolve_llm(
    provider: str, model: str | None, temperature: float
) -> tuple[str, Any | None]:
    """Normalise the provider name and build the LLM (exits on bad config)."""
    provider = provider.strip().lower().replace("_", "-")
    if provider not in PROVIDERS:
        console.print(
            f"[red]Unknown provider '{provider}'. Supported: {', '.join(PROVIDERS)}[/red]"
        )
        raise typer.Exit(code=2)

    if provider == "none":
        console.print(
            "[yellow]Offline mode (provider 'none'): deterministic heuristics, no LLM calls.[/yellow]"
        )
        return provider, None
    try:
        llm = get_llm(provider, model=model, temperature=temperature)
    except Exception as exc:
        console.print(f"[red]Cannot initialise provider '{provider}': {exc}[/red]")
        console.print(
            "[dim]Run [bold]sift providers[/bold] to see required configuration.[/dim]"
        )
        raise typer.Exit(code=2)
    shown = (
        model
        or getattr(llm, "model_name", None)
        or getattr(llm, "model", None)
        or DEFAULT_MODELS.get(provider)
        or "?"
    )
    console.print(f"[bold]LLM:[/bold] {provider} · {shown}")
    return provider, llm


@app.command()
def run(
    input_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            readable=True,
            help="Path to the CSV, Excel (.xlsx/.xlsm/.xls) or ODS (.ods) file to analyse.",
        ),
    ],
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            "-p",
            help=f"LLM provider: {', '.join(PROVIDERS)}. 'none' = offline heuristics.",
        ),
    ] = "none",
    model: Annotated[
        str | None,
        typer.Option(
            "--model", "-m", help="Model / deployment name (provider-specific)."
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output directory (default: output/<name>_<timestamp>).",
        ),
    ] = None,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature", "-t", min=0.0, max=2.0, help="LLM sampling temperature."
        ),
    ] = 0.2,
    sheet: Annotated[
        str | None,
        typer.Option(
            "--sheet",
            "-s",
            help=(
                "Excel/ODS sheet to load: name or 0-based index. "
                "Default: combine all non-empty sheets that share the same columns."
            ),
        ),
    ] = None,
) -> None:
    """Run the auto-EDA pipeline on a CSV/Excel/ODS file: stats -> charts -> insights -> recommendations -> report."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    provider, llm = _resolve_llm(provider, model, temperature)

    auto_dir = output is None
    if output is not None:
        if output.exists() and (output.is_file() or any(output.iterdir())):
            console.print(
                f"[red]Output path {output} already exists and is not empty — "
                "choose a new/empty directory so previous results are not overwritten.[/red]"
            )
            raise typer.Exit(code=2)
        out_dir = output
    else:
        base = Path("output") / f"{input_path.stem}_{datetime.now(UTC):%Y%m%d-%H%M%S}"
        out_dir = unique_dir(base)
    initial: dict = {
        "input_path": str(input_path),
        "sheet": sheet or "",
        "output_dir": str(out_dir),
        "provider": provider,
        "model": model or "",
    }

    final = run_pipeline(
        initial,
        llm,
        on_node=lambda node: console.print(f"  [green]✓[/green] [dim]{node}[/dim]"),
    )

    if final.get("error"):
        console.print(f"\n[red]Error:[/red] {final['error']}")
        if auto_dir:
            shutil.rmtree(out_dir, ignore_errors=True)
            console.print(f"[dim]Removed incomplete output directory {out_dir}[/dim]")
        raise typer.Exit(code=1)

    for w in final.get("warnings") or []:
        console.print(f"  [yellow]![/yellow] [dim]{w}[/dim]")

    health = ((final.get("profile") or {}).get("health")) or {}
    if health:
        grade = health.get("grade", "?")
        color = (
            "green"
            if health.get("score", 0) >= 90
            else ("yellow" if health.get("score", 0) >= 70 else "red")
        )
        console.print(
            f"[bold]Data health:[/bold] [{color}]{health.get('score')}/100 "
            f"(grade {grade})[/{color}] — {health.get('verdict', '')}"
        )

    console.rule("[bold]Key insights")
    console.print(Markdown(final.get("insights") or "_none_"))
    console.rule("[bold]Recommendations")
    console.print(Markdown(final.get("recommendations") or "_none_"))
    console.print()
    console.print(f"[bold green]Report:[/bold green]   {final.get('report_path')}")
    html_report = final.get("html_report_path")
    if html_report:
        console.print(f"[bold green]HTML:[/bold green]     {html_report}")
    console.print(
        f"[bold green]Charts:[/bold green]   {len(final.get('charts') or [])} file(s) in {Path(out_dir) / 'charts'}"
    )
    console.print(
        f"[bold green]Profile:[/bold green]  {Path(out_dir) / 'profile.json'}"
    )
    if provider == "none":
        console.print(
            "[dim]Tip: run [bold]sift providers[/bold] to see how to enable LLM-powered insights.[/dim]"
        )


@app.command()
def ask(
    input_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            readable=True,
            help="Path to the CSV, Excel (.xlsx/.xlsm/.xls) or ODS (.ods) file to ask about.",
        ),
    ],
    question: Annotated[
        str, typer.Argument(help="Natural-language question about the dataset.")
    ],
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            "-p",
            help=f"LLM provider: {', '.join(PROVIDERS)}. 'none' = offline intent engine.",
        ),
    ] = "none",
    model: Annotated[
        str | None,
        typer.Option(
            "--model", "-m", help="Model / deployment name (provider-specific)."
        ),
    ] = None,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature", "-t", min=0.0, max=2.0, help="LLM sampling temperature."
        ),
    ] = 0.2,
    sheet: Annotated[
        str | None,
        typer.Option(
            "--sheet",
            "-s",
            help=(
                "Excel/ODS sheet to load: name or 0-based index. "
                "Default: combine all non-empty sheets that share the same columns."
            ),
        ),
    ] = None,
) -> None:
    """Ask a natural-language question about a CSV/Excel/ODS file (offline by default)."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    _, llm = _resolve_llm(provider, model, temperature)

    try:
        df, _ = load_table(input_path, sheet=sheet)
    except LoadError as exc:
        console.print(f"\n[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"\n[red]Error:[/red] could not load '{input_path}': {exc}")
        raise typer.Exit(code=1)
    profile = profile_dataframe(df)

    result = answer_question(question, df, profile, llm=llm)
    console.rule("[bold]Answer")
    console.print(Markdown(result["answer"]))
    mode = "offline (deterministic)" if result["offline"] else "online (LLM)"
    console.print(f"[dim]mode: {mode}[/dim]")
    if result.get("sources"):
        console.print(f"[dim]sources: {', '.join(result['sources'])}[/dim]")
    if result.get("warning"):
        console.print(f"[yellow]![/yellow] [dim]{result['warning']}[/dim]")


@app.command()
def providers() -> None:
    """List supported LLM providers, required configuration and default models."""
    table = Table(title="sift — LLM providers")
    table.add_column("provider", style="bold")
    table.add_column("required configuration")
    table.add_column("default model", style="dim")
    for name in PROVIDERS:
        table.add_row(
            name, REQUIRED_ENV.get(name, ""), DEFAULT_MODELS.get(name, "") or "—"
        )
    console.print(table)
    console.print("[bold]Examples[/bold]")
    console.print("  sift run data.csv -p openai")
    console.print("  sift run data.csv -p anthropic -m claude-sonnet-4-5")
    console.print("  sift run data.csv -p azure -m my-gpt4o-deployment")
    console.print(
        "  sift run data.csv -p bedrock -m anthropic.claude-3-5-sonnet-20241022-v2:0"
    )
    console.print(
        "  OPENAI_BASE_URL=http://localhost:11434/v1 sift run data.csv -p openai-compatible -m qwen2.5:14b"
    )
    console.print('  sift ask data.csv "how many rows are missing col_x?" -p openai')


if __name__ == "__main__":
    app()
