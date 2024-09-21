import datetime
from rich.logging import RichHandler
from rich.console import Console
import logging

# Record the start time
start_time = datetime.datetime.now()

# Create a console for rich print
console = Console()

# Set up Rich logging configuration
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)


def log_info(message):
    current_time = datetime.datetime.now()
    elapsed = current_time - start_time
    # Add Rich markup for coloring and styling
    console.log(
        f"[green]{current_time:%Y-%m-%d %H:%M:%S}[/green] [bold blue]{message}[/bold blue] - Elapsed Time: [yellow]{elapsed}[/yellow]"
    )
