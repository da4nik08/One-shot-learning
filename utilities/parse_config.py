import argparse
from utilities import load_config


def parse_configs(CONFIGS_PATH):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="List of configuration file paths"
    )
    args = parser.parse_args()
    
    configs = {}
    for config_name in args.configs:
        config_key = config_name.replace(".yaml", "")
        configs[config_key] = load_config(CONFIGS_PATH, config_name)
    
    return configs