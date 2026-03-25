#!/usr/bin/env python3
"""
Audio Splitter — Separate vocals from instrumentals using Meta's Demucs.
A simple CLI with a clean TUI, inspired by lalal.ai.
"""

import sys
import os
import time
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text
from rich.table import Table
from rich import box

console = Console()

SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff"}
OUTPUT_DIR = Path.home() / "Documents" / "separated-audios"


def print_banner():
    banner = Text()
    banner.append("  audio", style="bold cyan")
    banner.append("splitter", style="bold magenta")
    banner.append("  ♪", style="bold yellow")
    console.print()
    console.print(Panel(
        banner,
        subtitle="[dim]powered by Meta's Demucs[/dim]",
        box=box.ROUNDED,
        padding=(1, 4),
    ))
    console.print()


def validate_file(filepath: str) -> Path:
    path = Path(filepath).expanduser().resolve()
    if not path.exists():
        console.print(f"  [red]Error:[/red] File not found: {path}")
        sys.exit(1)
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        console.print(f"  [red]Error:[/red] Unsupported format: {path.suffix}")
        console.print(f"  [dim]Supported: {', '.join(sorted(SUPPORTED_FORMATS))}[/dim]")
        sys.exit(1)
    return path


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA GPU"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon (MPS)"
    return torch.device("cpu"), "CPU"


def separate(audio_path: Path):
    song_name = audio_path.stem

    # Show file info
    size_mb = audio_path.stat().st_size / (1024 * 1024)
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column(style="dim")
    info_table.add_column(style="bold")
    info_table.add_row("File", audio_path.name)
    info_table.add_row("Size", f"{size_mb:.1f} MB")
    info_table.add_row("Format", audio_path.suffix.upper().lstrip("."))
    console.print(Panel(info_table, title="[bold]Input[/bold]", box=box.ROUNDED, padding=(1, 2)))
    console.print()

    device, device_name = get_device()
    console.print(f"  [dim]Device: {device_name}[/dim]\n")

    # Load model
    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task("Loading Demucs model (htdemucs)...", total=None)
        model = get_model("htdemucs")
        model.to(device)
        model.eval()

    console.print("  [green]✓[/green] Model loaded\n")

    # Load audio
    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task("Loading audio file...", total=None)
        wav, sr = torchaudio.load(str(audio_path))

        # Resample if needed
        if sr != model.samplerate:
            wav = torchaudio.transforms.Resample(sr, model.samplerate)(wav)

        # Ensure stereo
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]

    console.print("  [green]✓[/green] Audio loaded\n")

    # Separate
    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold]{task.description}"),
        TimeElapsedColumn(),
        transient=False,
        console=console,
    ) as progress:
        task = progress.add_task("Separating audio — this may take a minute...", total=None)
        start_time = time.time()

        with torch.no_grad():
            sources = apply_model(model, wav.unsqueeze(0).to(device))
        sources = sources[0]  # remove batch dim → (num_sources, channels, samples)

        elapsed = time.time() - start_time
        progress.update(task, description=f"[green]✓[/green] Separation complete — {elapsed:.1f}s")

    console.print()

    # Extract vocals and sum everything else as instrumental
    source_names = model.sources  # ['drums', 'bass', 'other', 'vocals']
    vocals_idx = source_names.index("vocals")
    vocals = sources[vocals_idx].cpu()

    instrumental = torch.zeros_like(vocals)
    for i, name in enumerate(source_names):
        if name != "vocals":
            instrumental += sources[i].cpu()

    # Save
    output_dir = OUTPUT_DIR / song_name
    output_dir.mkdir(parents=True, exist_ok=True)

    vocals_path = output_dir / f"{song_name}_vocals.wav"
    instrumental_path = output_dir / f"{song_name}_instrumental.wav"

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task("Saving audio files...", total=None)
        torchaudio.save(str(vocals_path), vocals, model.samplerate)
        torchaudio.save(str(instrumental_path), instrumental, model.samplerate)

    console.print("  [green]✓[/green] Files saved\n")

    # Output summary
    v_size = vocals_path.stat().st_size / (1024 * 1024)
    i_size = instrumental_path.stat().st_size / (1024 * 1024)

    out_table = Table(show_header=False, box=None, padding=(0, 2))
    out_table.add_column(style="dim")
    out_table.add_column(style="bold")
    out_table.add_row("Vocals", f"{vocals_path}")
    out_table.add_row("", f"[dim]{v_size:.1f} MB[/dim]")
    out_table.add_row("Instrumental", f"{instrumental_path}")
    out_table.add_row("", f"[dim]{i_size:.1f} MB[/dim]")

    console.print(Panel(out_table, title="[bold green]Output[/bold green]", box=box.ROUNDED, padding=(1, 2)))
    console.print()
    console.print(f"  [dim]Saved to: {output_dir}[/dim]\n")


def main():
    parser = argparse.ArgumentParser(
        description="Split audio into vocals and instrumentals using Demucs.",
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Path to audio file (mp3, wav, flac, ogg, m4a, aac, wma, aiff)",
    )
    args = parser.parse_args()

    print_banner()

    # If no file arg, prompt for it
    if args.file:
        filepath = args.file
    else:
        console.print("  [bold]Drop your audio file path here:[/bold]")
        console.print()
        try:
            filepath = console.input("  [cyan]>[/cyan] ").strip().strip("'\"")
        except (KeyboardInterrupt, EOFError):
            console.print("\n  [dim]Cancelled.[/dim]")
            sys.exit(0)

    if not filepath:
        console.print("  [red]No file provided.[/red]")
        sys.exit(1)

    audio_path = validate_file(filepath)
    console.print()

    try:
        separate(audio_path)
    except KeyboardInterrupt:
        console.print("\n  [yellow]Interrupted.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n  [red]Error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
