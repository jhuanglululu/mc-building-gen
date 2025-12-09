from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
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
    args = parser.parse_args()

    # train(args.config_root, args.config_name)


if __name__ == '__main__':
    main()
