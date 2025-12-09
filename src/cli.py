from argparse import ArgumentParser as _ArgumentParser


def get_arg_parser() -> _ArgumentParser:
    parser = _ArgumentParser()
    parser.add_argument(
        '--blocks', '-b', type=str, required=True, help='Block list dir'
    )
    parser.add_argument(
        '--config-name', type=str, required=True, help='Model config names'
    )
    parser.add_argument(
        '--config-root',
        type=str,
        default='config',
        help='Config folder path. Default: config',
    )
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument('--cpu', action='store_true', help='Use CPU')
    device_group.add_argument(
        '--mps', action='store_true', help='Use MPS (Apple Silicon)'
    )
    device_group.add_argument('--cuda', action='store_true', help='Use CUDA')

    return parser
