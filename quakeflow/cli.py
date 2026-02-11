"""
Command-line interface for QuakeFlow.
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import List, Optional, Tuple
import typer
import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from .config import Config
from .core.template_creator import TemplateCreator
from .core.template_matcher import TemplateMatcher
from .core.evaluator import ResultsEvaluator
from .utils.plotting import plot_templates_tsne, plot_templates_tsne_with_mapping


console = Console()
app = typer.Typer(help="QuakeFlow: Comprehensive template matching pipeline", rich_markup_mode="rich")


@app.command()
def init(
    config_file: Path = typer.Option(
        "config.yaml",
        "--config", "-c",
        help="Path to configuration file",
        show_default=True
    ),
    base_dir: Path = typer.Option(
        None,
        "--base-dir",
        help="Base directory for all outputs"
    ),
    station: str = typer.Option(
        None,
        "--station",
        help="Station code (e.g., GWBD)"
    )
):
    """Initialize a new configuration file."""
    
    config = Config()
    
    if base_dir:
        config.update('paths.base_dir', str(base_dir))
    if station:
        config.update('stations.station_code', station)
    
    config.save(config_file)
    
    console.print(Panel.fit(
        f"✅ Configuration file created: [bold cyan]{config_file}[/bold cyan]\n"
        f"📁 Please edit the file to adjust settings before running the pipeline.",
        title="QuakeFlow Initialization",
        border_style="green"
    ))
    
    # Show example config
    console.print("\n📋 Example usage:")
    console.print("  quakeflow create --config config.yaml --catalog catalog.csv --bbox 50.0 50.2 7.8 8.0")
    console.print("  quakeflow match --config config.yaml")
    console.print("  quakeflow evaluate --config config.yaml")
    console.print("  quakeflow run --config config.yaml --catalog catalog.csv --bbox 50.0 50.2 7.8 8.0")


@app.command()
def create(
    config_file: Path = typer.Option(
        "config.yaml",
        "--config", "-c",
        help="Path to configuration file"
    ),
    catalog: Optional[Path] = typer.Argument(
        None,
        help="Path to catalog file (CSV or DLF/.dat whitespace format). If omitted, uses `paths.catalog_file` from config."
    ),
    bbox: Optional[Tuple[float, float, float, float]] = typer.Option(
        None,
        "--bbox",
        help="Bounding box: provide four floats: min_lat max_lat min_lon max_lon (space-separated)",
    ),
    region: Optional[str] = typer.Option(
        None,
        "--region",
        help="Region name to filter events"
    ),
    catalog_type: str = typer.Option(
        "generic",
        "--type", "-t",
        help="Catalog type: 'grun', 'generic', or DLF-style 'dlf'/'dat'",
        show_default=True
    ),
    plot_tsne: bool = typer.Option(
        False,
        "--plot-tsne",
        help="Generate TSNE clustering visualization for templates"
    )
):
    """Create templates from catalog events."""
    
    if not config_file.exists():
        console.print(f"[red]Configuration file not found: {config_file}[/red]")
        console.print("Run 'quakeflow init' first to create a configuration file.")
        raise typer.Exit(1)

    # Load configuration early so we can fallback to `paths.catalog_file`
    config = Config(config_file)

    # Determine catalog: CLI argument takes precedence; otherwise use config.paths.catalog_file
    if catalog is None:
        cfg_catalog = config.get('paths.catalog_file')
        if cfg_catalog:
            catalog = Path(cfg_catalog)
        else:
            console.print(f"[red]Catalog file not provided and `paths.catalog_file` not set in config.[/red]")
            console.print("Provide a catalog on the command line or set `paths.catalog_file` in your config.")
            raise typer.Exit(1)

    if not catalog.exists():
        console.print(f"[red]Catalog file not found: {catalog}[/red]")
        raise typer.Exit(1)
    
    # Parse bounding box: expect a tuple of four floats from Typer. If not provided use default.
    if bbox is not None:
        # Typer will have already converted to a tuple of floats
        bbox_tuple = tuple(bbox)
        if len(bbox_tuple) != 4:
            console.print("[red]Error: --bbox requires exactly four numeric values[/red]")
            raise typer.Exit(1)
    else:
        console.print("[yellow]Using default bounding box (adjust in config if needed)[/yellow]")
        bbox_tuple = (49.0, 67, 5, 9.0)
    
    console.print(Panel.fit(
        f"[bold]Template Creation[/bold]\n"
        f"📁 Catalog: [cyan]{catalog}[/cyan]\n"
        f"📍 BBox: {bbox_tuple}\n"
        f"📍  Region: {region}\n"
        f"🏷️  Type: {catalog_type}",
        border_style="blue"
    ))
    
    # Create template creator
    creator = TemplateCreator(config)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        result = creator.create_templates(catalog, catalog_type, bbox_tuple, region, progress)
    
    if result["success"]:
        console.print(Panel.fit(
            f"✅ [bold green]Template creation successful![/bold green]\n"
            f"📁 Created {result['templates_created']} templates\n"
            f"💾 Saved to: {result.get('info_file', 'N/A')}",
            border_style="green"
        ))
        # Optional TSNE visualization
        if plot_tsne:
            try:
                plots_dir = config.get_path('plots_dir')
                templates_dir = result.get('templates_dir', config.get_path('templates_dir'))
                console.print("[dim]Generating t-SNE template visualization...[/dim]")
                plot_templates_tsne(templates_dir, plots_dir,
                                    eps=config.get('template_matching.cluster_eps', 0.2))
                console.print(Panel.fit(
                    f"🖼️ TSNE plot saved: [cyan]{plots_dir / 'templates_tsne.png'}[/cyan]",
                    border_style="cyan"
                ))
            except Exception as e:
                console.print(f"[yellow]TSNE visualization failed: {e}[/yellow]")
    else:
        console.print(Panel.fit(
            "[red]Template creation failed![/red]",
            border_style="red"
        ))
        raise typer.Exit(1)


@app.command()
def match(
    config_file: Path = typer.Option(
        "config.yaml",
        "--config", "-c",
        help="Path to configuration file"
    ),
    bbox: Optional[Tuple[float, float, float, float]] = typer.Option(
        None,
        "--bbox",
        help="Bounding box: provide four floats: min_lat max_lat min_lon max_lon (space-separated)",
    ),
    region: Optional[str] = typer.Option(
        None,
        "--region",
        help="Region name to filter templates/events",
    ),
    templates_dir: Optional[Path] = typer.Option(
        None,
        "--templates-dir",
        help="Directory containing templates (overrides config)"
    ),
    plot_tsne: bool = typer.Option(
        False,
        "--plot-tsne",
        help="Generate TSNE clustering visualization after matching"
    ),
    ignore_template_settings: bool = typer.Option(
        False,
        "--ignore-template-settings",
        help="Ignore template_processing.yaml settings in templates dir"
    )
):
    """Run template matching on continuous data."""
    
    if not config_file.exists():
        console.print(f"[red]Configuration file not found: {config_file}[/red]")
        raise typer.Exit(1)
    
    # Load configuration
    config = Config(config_file)
    
    if templates_dir:
        config.update('paths.templates_dir', str(templates_dir))
    
    console.print(Panel.fit(
        f"[bold]Template Matching[/bold]\n"
        f"🏢 Station: [cyan]{config['stations.station_code']}[/cyan]\n"
        f"📡 Channel: [cyan]{config['stations.primary_channel']}[/cyan]\n"
        f"📅 Start: [cyan]{config['template_matching.start_date']}[/cyan]\n"
        f"📊 Days: [cyan]{config['template_matching.days_to_process']}[/cyan]",
        border_style="blue"
    ))
    
    # Create template matcher
    matcher = TemplateMatcher(config)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        result = matcher.match_templates(
            config.get_path('templates_dir'),
            progress,
            plot_tsne_pre=plot_tsne,
            ignore_template_settings=ignore_template_settings,
            bbox=bbox,
            region=region,
        )
    
    if result["success"]:
        console.print(Panel.fit(
            f"✅ [bold green]Template matching successful![/bold green]\n"
            f"📅 Processed: {result['days_processed']} days\n"
            f"✅ Successful: {result['successful_days']} days\n"
            f"📁 Results in: {result['output_dir']}",
            border_style="green"
        ))
        if plot_tsne:
            try:
                plots_dir = config.get_path('plots_dir')
                tdir = templates_dir or config.get_path('templates_dir')
                mapping_file = config.get_path('base_dir') / config.get_path('templates_dir') / "template_id_mapping.csv"
                console.print("[dim]Generating t-SNE visualization with matcher mapping...[/dim]")
                plot_templates_tsne_with_mapping(tdir, mapping_file, plots_dir)
                console.print(Panel.fit(
                    f"🖼️ TSNE plot saved: [cyan]{plots_dir / 'templates_tsne_match.png'}[/cyan]",
                    border_style="cyan"
                ))
            except Exception as e:
                console.print(f"[yellow]TSNE visualization failed: {e}[/yellow]")
    else:
        console.print(Panel.fit(
            "[red]Template matching failed![/red]",
            border_style="red"
        ))
        raise typer.Exit(1)


@app.command()
def evaluate(
    config_file: Path = typer.Option(
        "config.yaml",
        "--config", "-c",
        help="Path to configuration file"
    ),
    detection_dir: Optional[Path] = typer.Option(
        None,
        "--detection-dir",
        help="Directory containing detection files (overrides config)"
    ),
    template_info: Optional[Path] = typer.Option(
        None,
        "--template-info",
        help="Template info CSV file (overrides config)"
    ),
    min_similarity: float = typer.Option(
        0.0,
        "--min-similarity",
        help="Minimum similarity threshold to consider a detection") 
    ):
    """Evaluate template matching results."""
    
    if not config_file.exists():
        console.print(f"[red]Configuration file not found: {config_file}[/red]")
        raise typer.Exit(1)
    
    # Load configuration
    config = Config(config_file)
    
    # Determine detection directory
    if detection_dir:
        det_dir = detection_dir
    else:
        station = config['stations.station_code']
        channel = config['stations.primary_channel']
        det_dir = config.get_path('base_dir') / "similarity" / f"{station}_{channel}"
    
    # Determine template info file
    if template_info:
        tpl_info = template_info
    else:
        tpl_info = config.get_path('templates_dir') / config['paths.template_info_file']
    
    console.print(Panel.fit(
        f"[bold]Results Evaluation[/bold]\n"
        f"📁 Detections: [cyan]{det_dir}[/cyan]\n"
        f"📋 Templates: [cyan]{tpl_info}[/cyan]",
        border_style="blue"
    ))
    
    if not det_dir.exists():
        console.print(f"[red]Detection directory not found: {det_dir}[/red]")
        raise typer.Exit(1)
    
    if not tpl_info.exists():
        console.print(f"[red]Template info file not found: {tpl_info}[/red]")
        raise typer.Exit(1)
    
    # Create evaluator
    evaluator = ResultsEvaluator(config)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        result = evaluator.evaluate(det_dir, tpl_info, progress, min_similarity)
    
    if result["success"]:
        console.print(Panel.fit(
            f"✅ [bold green]Evaluation successful![/bold green]\n"
            f"📊 Detections: {result['detections']:,}\n"
            f"⚡ b-value: {result['b_value']:.2f}\n"
            f"📁 Results: {result['output_file']}",
            border_style="green"
        ))
    else:
        console.print(Panel.fit(
            "[red]Evaluation failed![/red]",
            border_style="red"
        ))
        raise typer.Exit(1)


@app.command()
def run(
    config_file: Path = typer.Option(
        "config.yaml",
        "--config", "-c",
        help="Path to configuration file"
    ),
    catalog: Optional[Path] = typer.Argument(
        None,
        help="Path to catalog file (CSV or DLF/.dat whitespace format). If omitted, uses `paths.catalog_file` from config."
    ),
    bbox: Optional[Tuple[float, float, float, float]] = typer.Option(
        None,
        "--bbox",
        help="Bounding box: provide four floats: min_lat max_lat min_lon max_lon (space-separated)",
    ),
    catalog_type: str = typer.Option(
        "generic",
        "--type", "-t",
        help="Catalog type: 'grun', 'generic', or DLF-style 'dlf'/'dat'",
        show_default=True
    ),
    skip_create: bool = typer.Option(
        False,
        "--skip-create",
        help="Skip template creation (use existing templates)"
    ),
    skip_match: bool = typer.Option(
        False,
        "--skip-match",
        help="Skip template matching (use existing detections)"
    ),
    skip_evaluate: bool = typer.Option(
        False,
        "--skip-evaluate",
        help="Skip evaluation"
    )
    ,
    region: Optional[str] = typer.Option(
        None,
        "--region",
        help="Region name to filter events/templates when running",
    )
):
    """Run the complete pipeline: create → match → evaluate."""
    
    # Ensure configuration file exists and load it before we attempt to use it
    if not config_file.exists():
        console.print(f"[red]Configuration file not found: {config_file}[/red]")
        console.print("Run 'quakeflow init' first to create a configuration file.")
        raise typer.Exit(1)

    # Load configuration
    config = Config(config_file)

    # Determine catalog (fallback to config) and parse bounding box
    # Only require a catalog if we're going to run creation (i.e., not skip_create)
    if not skip_create:
        if catalog is None:
            cfg_catalog = config.get('paths.catalog_file')
            if cfg_catalog:
                catalog = Path(cfg_catalog)
            else:
                console.print(f"[red]Catalog file not provided and `paths.catalog_file` not set in config.[/red]")
                console.print("Provide a catalog on the command line or set `paths.catalog_file` in your config.")
                raise typer.Exit(1)
        if not catalog.exists():
            console.print(f"[red]Catalog file not found: {catalog}[/red]")
            raise typer.Exit(1)

    # Parse bounding box: prefer tuple of floats from Typer, otherwise default
    if bbox is not None:
        bbox_tuple = tuple(bbox)
        if len(bbox_tuple) != 4:
            console.print("[red]Error: --bbox requires exactly four numeric values[/red]")
            raise typer.Exit(1)
    else:
        console.print("[yellow]Using default bounding box[/yellow]")
        bbox_tuple = (50.0, 50.2, 7.8, 8.0)
    
    # Rest of your function...
    
    console.print(Panel.fit(
        "[bold]QuakeFlow[/bold] - Complete Pipeline\n"
        "A comprehensive template matching system",
        border_style="cyan"
    ))
    
    if not config_file.exists():
        console.print(f"[red]Configuration file not found: {config_file}[/red]")
        raise typer.Exit(1)
    
    # Load configuration
    config = Config(config_file)

    # Determine catalog: CLI argument takes precedence; otherwise use config.paths.catalog_file
    if catalog is None and not skip_create:
        cfg_catalog = config.get('paths.catalog_file')
        if cfg_catalog:
            catalog = Path(cfg_catalog)
        else:
            console.print(f"[red]Catalog file not provided and `paths.catalog_file` not set in config.[/red]")
            console.print("Provide a catalog on the command line or set `paths.catalog_file` in your config.")
            raise typer.Exit(1)

    # `bbox_tuple` is parsed above from `bbox_str`; ensure it exists and fallback if not
    if 'bbox_tuple' not in locals():
        console.print("[yellow]Using default bounding box[/yellow]")
        bbox_tuple = (50.0, 50.2, 7.8, 8.0)
    
    results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        # Step 1: Create templates
        if not skip_create:
            console.print("\n[bold cyan]Step 1: Creating Templates[/bold cyan]")
            creator = TemplateCreator(config)
            # `create_templates` signature: (catalog_path, catalog_type, bbox, region=None, progress=None)
            # In `run` we don't have a `region` argument, so pass it explicitly as None and provide
            # the `progress` kwarg to avoid accidental positional mapping of progress -> region.
            results['create'] = creator.create_templates(
                catalog,
                catalog_type,
                bbox_tuple,
                region=None,
                progress=progress,
            )
            
            if not results['create']['success']:
                console.print("[red]Template creation failed![/red]")
                raise typer.Exit(1)
        
        # Step 2: Template matching
        if not skip_match:
            console.print("\n[bold cyan]Step 2: Template Matching[/bold cyan]")
            matcher = TemplateMatcher(config)
            results['match'] = matcher.match_templates(
                config.get_path('templates_dir'),
                progress,
                bbox=bbox_tuple,
                region=region,
            )
            
            if not results['match']['success']:
                console.print("[yellow]Template matching had issues, but continuing...[/yellow]")
        
        # Step 3: Evaluation
        if not skip_evaluate:
            console.print("\n[bold cyan]Step 3: Results Evaluation[/bold cyan]")
            
            # Determine paths
            station = config['stations.station_code']
            channel = config['stations.primary_channel']
            det_dir = config.get_path('base_dir') / "similarity" / f"{station}_{channel}"
            tpl_info = config.get_path('templates_dir') / config['paths.template_info_file']
            
            evaluator = ResultsEvaluator(config)
            results['evaluate'] = evaluator.evaluate(det_dir, tpl_info, progress)
    
    # Final summary
    console.print(Panel.fit(
        "[bold green]✅ Pipeline Complete![/bold green]\n\n"
        f"📊 [bold]Summary:[/bold]\n"
        f"   📁 Templates created: {results.get('create', {}).get('templates_created', 'Skipped')}\n"
        f"   📅 Days processed: {results.get('match', {}).get('days_processed', 'Skipped')}\n"
        f"   📈 b-value: {results.get('evaluate', {}).get('b_value', 'Skipped'):.2f}\n"
        f"   ⚡ Mc: {results.get('evaluate', {}).get('mc', 'Skipped'):.2f}",
        border_style="green"
    ))


@app.command()
def status(
    config_file: Path = typer.Option(
        "config.yaml",
        "--config", "-c",
        help="Path to configuration file"
    )
):
    """Show pipeline status and statistics."""
    
    if not config_file.exists():
        console.print(f"[red]Configuration file not found: {config_file}[/red]")
        raise typer.Exit(1)
    
    config = Config(config_file)
    
    # Create status table
    table = Table(title="QuakeFlow Pipeline Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="white")
    
    # Check templates
    templates_dir = config.get_path('templates_dir')
    template_files = list(templates_dir.glob("*.mseed")) if templates_dir.exists() else []
    table.add_row(
        "Templates",
        "✅ Found" if template_files else "❌ Missing",
        f"{len(template_files)} files" if template_files else "Not created"
    )
    
    # Check template info
    info_file = templates_dir / config['paths.template_info_file']
    table.add_row(
        "Template Info",
        "✅ Found" if info_file.exists() else "❌ Missing",
        str(info_file) if info_file.exists() else "Not found"
    )
    
    # Check detections
    station = config['stations.station_code']
    channel = config['stations.primary_channel']
    det_dir = config.get_path('base_dir') / "similarity" / f"{station}_{channel}"
    det_files = list(det_dir.glob("detections_*.csv")) if det_dir.exists() else []
    table.add_row(
        "Detections",
        "✅ Found" if det_files else "❌ Missing",
        f"{len(det_files)} files" if det_files else "Not created"
    )
    
    # Check results
    output_dir = config.get_path('output_dir')
    results_file = output_dir / "detections_with_mag.csv"
    table.add_row(
        "Results",
        "✅ Found" if results_file.exists() else "❌ Missing",
        str(results_file) if results_file.exists() else "Not evaluated"
    )
    
    console.print(table)
    
    # Show configuration summary
    console.print("\n[bold cyan]Configuration Summary:[/bold cyan]")
    console.print(f"  🏢 Station: {config['stations.station_code']}")
    console.print(f"  📍 Location: {config['stations.lat']:.3f}°N, {config['stations.lon']:.3f}°E")
    console.print(f"  📅 Start date: {config['template_matching.start_date']}")
    console.print(f"  📊 Days to process: {config['template_matching.days_to_process']}")


@app.command()
def multi_iterate(
    config_file: Path = typer.Option(
        "config.yaml",
        "--config", "-c",
        help="Path to configuration file"
    ),
    catalog: Optional[Path] = typer.Argument(
        None,
        help="Path to initial catalog file (CSV or DLF/.dat whitespace format). If omitted, uses `paths.catalog_file` from config."
    ),
    iterations: int = typer.Option(
        2,
        "--iterations", "-i",
        help="Number of detection iterations",
        show_default=True
    ),
    bbox: Optional[Tuple[float, float, float, float]] = typer.Option(
        None,
        "--bbox",
        help="Bounding box: provide four floats: min_lat max_lat min_lon max_lon (space-separated)",
    ),
    catalog_type: str = typer.Option(
        "generic",
        "--type", "-t",
        help="Catalog type: 'grun', 'generic', or DLF-style 'dlf'/'dat'",
        show_default=True
    ),
    region: Optional[str] = typer.Option(
        None,
        "--region",
        help="Region name to filter events"
    ),
    skip_evaluate: bool = typer.Option(
        False,
        "--skip-evaluate",
        help="Skip evaluation step"
    )
):
    """Run multi-iteration detection with catalog updates.
    
    Workflow:
    1. Create templates from initial catalog
    2. Run detection iteration 1
    3. Generate catalog from detections
    4. Create templates from new catalog
    5. Run detection iteration 2
    ... repeat steps 3-5 for each iteration
    """
    
    if not config_file.exists():
        console.print(f"[red]Configuration file not found: {config_file}[/red]")
        raise typer.Exit(1)

    # Load configuration early so we can fallback to `paths.catalog_file`
    config = Config(config_file)

    # Determine catalog: CLI takes precedence, fallback to config
    if catalog is None:
        cfg_catalog = config.get('paths.catalog_file')
        if cfg_catalog:
            catalog = Path(cfg_catalog)
        else:
            console.print(f"[red]Catalog file not provided and `paths.catalog_file` not set in config.[/red]")
            console.print("Provide a catalog on the command line or set `paths.catalog_file` in your config.")
            raise typer.Exit(1)

    if not catalog.exists():
        console.print(f"[red]Catalog file not found: {catalog}[/red]")
        raise typer.Exit(1)

    # Parse bounding box: prefer tuple of floats from Typer, otherwise default
    if bbox is not None:
        bbox_tuple = tuple(bbox)
        if len(bbox_tuple) != 4:
            console.print("[red]Error: --bbox requires exactly four numeric values[/red]")
            raise typer.Exit(1)
    else:
        console.print("[yellow]Using default bounding box[/yellow]")
        bbox_tuple = (50.0, 50.2, 7.8, 8.0)
    
    console.print(Panel.fit(
        "[bold]QuakeFlow Multi-Iteration Detection[/bold]\n"
        f"📊 Total iterations: [cyan]{iterations}[/cyan]\n"
        f"🏢 Station: [cyan]{config['stations.station_code']}[/cyan]\n"
        f"📡 Channel: [cyan]{config['stations.primary_channel']}[/cyan]",
        border_style="cyan"
    ))
    
    current_catalog = catalog
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        for iteration in range(1, iterations + 1):
            console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
            console.print(f"[bold cyan]Iteration {iteration}/{iterations}[/bold cyan]")
            console.print(f"[bold cyan]{'='*60}[/bold cyan]")
            
            # Step 1: Create templates
            console.print(f"\n[cyan]Step 1.{iteration}: Creating Templates[/cyan]")
            creator = TemplateCreator(config)
            result_create = creator.create_templates(
                current_catalog, catalog_type, bbox_tuple, region, progress
            )
            
            if not result_create['success']:
                console.print(f"[red]Template creation failed at iteration {iteration}![/red]")
                raise typer.Exit(1)
            
            console.print(f"  ✅ Created {result_create['templates_created']} templates")
            
            # Step 2: Template matching
            console.print(f"\n[cyan]Step 2.{iteration}: Template Matching[/cyan]")
            matcher = TemplateMatcher(config)
            result_match = matcher.match_templates(
                config.get_path('templates_dir'), progress
            )
            
            if not result_match['success']:
                console.print(f"[yellow]Template matching had issues at iteration {iteration}, but continuing...[/yellow]")
            
            console.print(f"  ✅ Processed {result_match['successful_days']} days")
            
            # Step 3: Evaluation and catalog generation
            if not skip_evaluate or iteration < iterations:
                console.print(f"\n[cyan]Step 3.{iteration}: Evaluation & Catalog Generation[/cyan]")
                
                station = config['stations.station_code']
                channel = config['stations.primary_channel']
                det_dir = config.get_path('base_dir') / "similarity" / f"{station}_{channel}"
                tpl_info = config.get_path('templates_dir') / config['paths.template_info_file']
                
                evaluator = ResultsEvaluator(config)
                result_eval = evaluator.evaluate(det_dir, tpl_info, progress)
                
                console.print(f"  ✅ b-value: {result_eval.get('b_value', np.nan):.2f}")
                console.print(f"  ✅ Mc: {result_eval.get('mc', np.nan):.2f}")
                
                # Generate updated catalog for next iteration
                if iteration < iterations:
                    console.print(f"\n[cyan]Generating catalog for iteration {iteration + 1}...[/cyan]")
                    
                    # Get the detections with magnitudes
                    output_dir = config.get_path('output_dir')
                    detections_file = output_dir / "detections_with_magnitude.csv"
                    
                    if detections_file.exists():
                        detections_df = pd.read_csv(detections_file)
                        
                        # Filter detections above magnitude threshold
                        min_mag = config.get('evaluation.min_magnitude', 0.0)
                        filtered_df = detections_df[detections_df['magnitude'] >= min_mag].copy()
                        
                        # Create new catalog with required columns
                        new_catalog_df = pd.DataFrame({
                            'time': pd.to_datetime(filtered_df['time']),
                            'lat': filtered_df['lat'],
                            'lon': filtered_df['lon'],
                            'magnitude': filtered_df['magnitude'],
                        })
                        
                        # Add depth if available
                        if 'depth' in filtered_df.columns:
                            new_catalog_df['depth'] = filtered_df['depth']
                        else:
                            new_catalog_df['depth'] = 10.0  # Default depth
                        
                        # Save updated catalog for next iteration
                        current_catalog = output_dir / f"catalog_iteration_{iteration}.csv"
                        new_catalog_df.to_csv(current_catalog, index=False)
                        
                        console.print(f"  ✅ Generated {len(new_catalog_df)} events for next iteration")
                        console.print(f"  📁 Saved to: [cyan]{current_catalog}[/cyan]")
                    else:
                        console.print(f"  [red]Detection file not found: {detections_file}[/red]")
                        raise typer.Exit(1)
    
    console.print(f"\n[bold green]✅ Multi-iteration detection complete![/bold green]")
    console.print(f"[cyan]Total iterations: {iterations}[/cyan]")
    console.print(f"[cyan]Final results in: {config.get_path('output_dir')}[/cyan]")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()