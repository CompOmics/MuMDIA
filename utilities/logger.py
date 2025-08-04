import datetime
import logging

from rich.console import Console
from rich.logging import RichHandler

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
    # Simple console logging without complex Rich markup
    simple_message = (
        f"{current_time:%Y-%m-%d %H:%M:%S} {message} - Elapsed Time: {elapsed}"
    )
    console.print(simple_message)
