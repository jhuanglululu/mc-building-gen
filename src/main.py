from rich.theme import Theme
from rich.console import Console
from rich.highlighter import RegexHighlighter, NullHighlighter
import torch
from rich.logging import RichHandler
import logging
from training import train  # pyright:ignore[reportUnknownVariableType]
from data import load_blocks
from cli import get_arg_parser

log = logging.getLogger('training')


class NumberHighlighter(RegexHighlighter):
    highlights = [
        r'(?P<keyword>(?:Epoch|Loss|Time|Total|ETA))',
        r'\b(?P<posnum>(?:\d+(?:\.\d*)?|\.\d+))',
        r'\d(?P<timeunit>(?:h|m|s)\b)',
    ]


def check_device():
    import torch

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device


def main():
    console = Console(
        theme=Theme(
            {'keyword': 'bold yellow', 'posnum': 'bright_green', 'timeunit': 'dim'}
        )
    )

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            RichHandler(
                console=console,
                highlighter=NumberHighlighter(),
                log_time_format='[%H:%M:%S]',
                rich_tracebacks=True,
                show_path=False,
            )
        ],
    )

    args = get_arg_parser().parse_args()

    if args.cpu:
        device = 'cpu'
    elif args.mps and torch.mps.is_available():
        device = 'mps'
    elif args.cuda and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = check_device()

    torch.device(device)
    log.info(f'Using device {device}')

    load_blocks(args.blocks)
    train(args.config_root, args.config_name, device)


if __name__ == '__main__':
    main()
