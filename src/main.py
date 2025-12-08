from config import VAETraningConfig, VAECheckpointConfig, VAEParameterConfig, DataConfig


def main():
    print(VAETraningConfig.read("config", "overfit.ini"))  # ty: ignore
    print(VAECheckpointConfig.read("config", "overfit.ini"))  # ty: ignore
    print(VAEParameterConfig.read("config", "overfit.ini"))  # ty: ignore
    print(DataConfig.read("config", "overfit.ini"))  # ty: ignore


if __name__ == "__main__":
    main()
