import argparse
from engine.engine import Engine

def main(args):
    engine = Engine(config_path=args.config)
    if engine.cfg["MODE"].lower() == "train":
        _, elapsed_time = engine.train()
        print(f"Training time: {elapsed_time:.2f} secs")
    elif engine.cfg["MODE"].lower() == "test":
        res, elapsed_time = engine.test()

        print(f"Testing time: {elapsed_time:.2f} secs")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')

    args = parser.parse_args()
    main(args)