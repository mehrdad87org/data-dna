import sys
import click
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from datadna.fingerprint import DataDNA

console = Console()


@click.group()
def main():
    """DataDNA - Synthetic data quality evaluation with DNA fingerprints."""
    pass


@main.command()
@click.argument("data_path")
@click.option("--output", "-o", default=None, help="Output JSON path")
def fingerprint(data_path, output):
    """Generate a DNA fingerprint for a dataset."""
    console.print(Panel.fit("[bold purple]DataDNA[/bold purple] Fingerprint Generator", border_style="purple"))

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        console.print(f"[red]Error loading data:[/red] {e}")
        sys.exit(1)

    dna = DataDNA(df, name=data_path)

    table = Table(title="DNA Fingerprint", box=box.ROUNDED, border_style="purple")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Dataset", data_path)
    table.add_row("Shape", f"{df.shape[0]} rows × {df.shape[1]} columns")
    table.add_row("DNA String", dna.dna_string)
    table.add_row("Column Entropy", str(dna.fingerprint.column_count_entropy))
    table.add_row("Mean Hash", dna.fingerprint.mean_distribution)
    table.add_row("Correlation Hash", dna.fingerprint.correlation_hash)
    table.add_row("Entropy Fingerprint", dna.fingerprint.entropy_fingerprint)

    console.print(table)

    if output:
        import json
        with open(output, "w") as f:
            json.dump(dna.to_dict(), f, indent=2, default=str)
        console.print(f"[green]Fingerprint saved to:[/green] {output}")


@main.command()
@click.argument("real_path")
@click.argument("synthetic_path")
@click.option("--output", "-o", default="datadna_report.html", help="Output HTML report")
def compare(real_path, synthetic_path, output):
    """Compare real vs synthetic dataset quality."""
    console.print(Panel.fit("[bold purple]DataDNA[/bold purple] Comparison", border_style="purple"))

    try:
        real_df = pd.read_csv(real_path)
        syn_df = pd.read_csv(synthetic_path)
    except Exception as e:
        console.print(f"[red]Error loading data:[/red] {e}")
        sys.exit(1)

    real_dna = DataDNA(real_df, name="real")
    syn_dna = DataDNA(syn_df, name="synthetic")

    comp = real_dna.compare(syn_dna)

    grade_color = "green" if comp.grade.startswith("A") else "yellow" if comp.grade.startswith("B") else "red"

    table = Table(title="Comparison Results", box=box.ROUNDED, border_style="purple")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Quality Score", f"[{grade_color}]{comp.quality_score}/100[/{grade_color}]")
    table.add_row("Grade", f"[{grade_color}]{comp.grade}[/{grade_color}]")
    table.add_row("ML Utility Score", f"{comp.ml_utility_score}/100")
    table.add_row("DNA Similarity", f"{comp.dna_similarity:.1%}")
    table.add_row("Mode Collapse", "⚠ DETECTED" if comp.mode_collapse else "✓ None")
    table.add_row("Real DNA", comp.real_dna)
    table.add_row("Synthetic DNA", comp.synthetic_dna)

    console.print(table)

    dim_table = Table(title="Dimension Scores", box=box.ROUNDED, border_style="purple")
    dim_table.add_column("Dimension", style="cyan")
    dim_table.add_column("Score", justify="right")

    for dim, score in sorted(comp.dimension_scores.items(), key=lambda x: -x[1]):
        pct = score * 100
        color = "green" if pct > 70 else "yellow" if pct > 50 else "red"
        dim_table.add_row(dim.replace("_", " ").title(), f"[{color}]{pct:.1f}%[/{color}]")

    console.print(dim_table)

    with console.status(f"[bold purple]Generating report: {output}[/bold purple]"):
        comp.generate_report(output)
        console.print(f"[green]Report saved to:[/green] {output}")


@main.command()
@click.argument("real_path")
@click.argument("synthetic_paths", nargs=-1)
@click.option("--output", "-o", default="datadna_benchmark.html", help="Output benchmark report")
def benchmark(real_path, synthetic_paths, output):
    """Benchmark multiple synthetic datasets against real data."""
    console.print(Panel.fit("[bold purple]DataDNA[/bold purple] Benchmark", border_style="purple"))

    try:
        real_df = pd.read_csv(real_path)
    except Exception as e:
        console.print(f"[red]Error loading real data:[/red] {e}")
        sys.exit(1)

    real_dna = DataDNA(real_df, name="real")
    results = []

    for syn_path in synthetic_paths:
        try:
            syn_df = pd.read_csv(syn_path)
            syn_dna = DataDNA(syn_df, name=syn_path)
            comp = real_dna.compare(syn_dna)
            results.append((syn_path, comp))
        except Exception as e:
            console.print(f"[yellow]Skipping {syn_path}: {e}[/yellow]")

    table = Table(title="Benchmark Results", box=box.ROUNDED, border_style="purple")
    table.add_column("Dataset", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Grade")
    table.add_column("ML Utility", justify="right")
    table.add_column("Mode Collapse")
    table.add_column("DNA Similarity", justify="right")

    for path, comp in sorted(results, key=lambda x: -x[1].quality_score):
        grade_color = "green" if comp.grade.startswith("A") else "yellow" if comp.grade.startswith("B") else "red"
        mc_color = "red" if comp.mode_collapse else "green"
        table.add_row(
            path,
            f"[{grade_color}]{comp.quality_score}/100[/{grade_color}]",
            f"[{grade_color}]{comp.grade}[/{grade_color}]",
            f"{comp.ml_utility_score}/100",
            f"[{mc_color}]{'Yes' if comp.mode_collapse else 'No'}[/{mc_color}]",
            f"{comp.dna_similarity:.1%}",
        )

    console.print(table)


if __name__ == "__main__":
    main()
