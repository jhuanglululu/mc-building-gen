from argparse import ArgumentParser
from train import train


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--config-name', type=str, required=True, help='Model config names'
    )
    parser.add_argument(
        '--config-root',
        type=str,
        default='config',
        help='Config folder path. Default: config',
    )
    args = parser.parse_args()

    train(args.config_root, args.config_name)


if __name__ == '__main__':
    main()
