import argparse
import copy
from pathlib import Path

import yaml


class ConfigKeyError(Exception):
    """Raised when a configuration key is not found."""

    pass


class Config:
    def __init__(self, config):
        self._config = {}
        for key, value in config.items():
            if isinstance(value, dict):
                self._config[key] = Config(value)
            else:
                self._config[key] = value

    def __getitem__(self, key):
        return self.get(key)

    # def __getattr__(self, key):
    #     try:
    #         return self.get(key)
    #     except ConfigKeyError as e:
    #         raise AttributeError(str(e))

    def __contains__(self, key):
        return key in self._config

    def get(self, key, default=ConfigKeyError):
        key_parts = key.split(".")
        obj = self
        for part in key_parts:
            if isinstance(obj, Config) and part in obj._config:
                obj = obj._config[part]
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                if default is ConfigKeyError:
                    raise ConfigKeyError(f"Configuration key '{key}' not found")
                return default
        return obj

    def update(self, key, val):
        key_parts = key.split(".")
        obj = self
        for part in key_parts[:-1]:
            if part not in obj._config:
                obj._config[part] = Config({})
            obj = obj._config[part]
        obj._config[key_parts[-1]] = val

    def to_dict(self):
        result = {}
        for key, value in self._config.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected for argument")


def process_arguments(default_conf=None, debug_conf=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default=None, required=False)
    parser.add_argument("-d", "--debug", action="store_true")
    args = vars(parser.parse_known_args()[0])
    if args["config"] is not None:
        assert args["debug"] is False
        conf = yaml.safe_load(args["config"].read_text())
    elif args["debug"]:
        assert debug_conf is not None
        conf = yaml.safe_load(Path(debug_conf).read_text())
    else:
        if default_conf is None:
            parser.error("--config is required")
        if isinstance(default_conf, str):
            conf = yaml.safe_load(Path(default_conf).read_text())
        else:
            conf = copy.deepcopy(default_conf)

    def call_dict(diction, args):
        args_list = args.split(".")
        for arg in args_list:
            diction = diction[arg]
        return diction

    def update_dict(diction, path, val):
        if len(path) > 1:
            update_dict(diction[path[0]], path[1:], val)
        else:
            diction[path[0]] = val
            return None

    args_to_create = []
    root = list(conf.keys())
    queue = [str(x) for x in root]
    visited = set()
    while len(queue) > 0:
        cur = queue.pop()
        if cur in visited:
            continue
        visited.add(cur)
        cur_call = call_dict(conf, cur)
        if isinstance(call_dict(conf, cur), dict):
            queue = queue + [str(cur) + "." + str(x) for x in list(cur_call.keys())]
        else:
            args_to_create.append(("--" + str(cur), cur_call))

    for key, val in args_to_create:
        if type(val) is bool:
            parser.add_argument(key, type=str2bool, required=False)
        else:
            if val == "__required__":
                parser.add_argument(key, required=True)
            elif val is None:
                parser.add_argument(key, required=False)
            else:
                parser.add_argument(key, type=type(val), required=False)

    args = vars(parser.parse_args())
    for key, _ in args_to_create:
        ckey = key[2:]
        if args[ckey] is not None:
            if isinstance(args[ckey], str) and args[ckey].isdigit():
                args[ckey] = int(args[ckey])
            update_dict(conf, ckey.split("."), args[ckey])
    return conf
