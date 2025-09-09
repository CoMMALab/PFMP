import argparse
import yaml
import os
import sys

CONFIG_DIR = "tests/problems"

def load_yaml(path: str) -> dict:
    """Load YAML file as a dictionary."""


    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data


def build_parser_and_parse_args(argv = sys.argv[1:]) -> argparse.Namespace:
    """
    Dynamically create an argparse parser with arguments
    matching YAML keys. These will be optional, allowing CLI overrides.
    """
    parser = argparse.ArgumentParser(description="PFMP Arguments")

    parser.add_argument("--config", type=str, default=os.path.join(CONFIG_DIR, "inverted_pendulum.yaml"),
                        help="Path to YAML config file")

    config_name = parser.parse_known_args(argv)[0].config
    yaml_data = load_yaml(config_name)

    for key, value in yaml_data.items():
        arg_type = type(value)
        if arg_type is bool:
            # Boolean flags
            parser.add_argument(f"--{key}", action="store_true" if not value else "store_false",
                                help=f"Boolean flag for {key} (default: {value})")
        else:
            parser.add_argument(f"--{key}", type=arg_type,
                                help=f"Override {key} (default: {value})", default = value)

    return parser.parse_known_args(argv)[0]


def test_argparse():

    args = build_parser_and_parse_args(["--K", "250"])
    print(args)
    args = build_parser_and_parse_args(["--config", os.path.join(CONFIG_DIR, "dummy.yaml"), "--L", "25"])
    print(args)


if __name__ == "__main__":
    test_argparse()