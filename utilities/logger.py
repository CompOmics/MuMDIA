import datetime

start_time = datetime.datetime.now()
from rich.logging import RichHandler
from rich.console import Console

console = Console()


def log_info(message):
    current_time = datetime.datetime.now()
    elapsed = current_time - start_time
    # Add Rich markup for coloring and styling
    console.log(
        f"[green]{current_time:%Y-%m-%d %H:%M:%S}[/green] [bold blue]{message}[/bold blue] - Elapsed Time: [yellow]{elapsed}[/yellow]"
    )
