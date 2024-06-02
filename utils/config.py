import json
from sklearn.utils import Bunch
import os


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(**config_dict)

    return config, config_dict


def process_config(json_file):
    import time
    config, _ = get_config_from_json(json_file)
    time_str = time.strftime("%Y%m%d-%H%M%S")
    exp_dir = os.path.join("experiments", f"{config.exp_name}-{time_str}")
    config.exp_dir = exp_dir
    config.summary_dir = os.path.join(exp_dir, "summary/")
    config.checkpoint_dir = os.path.join(exp_dir, "checkpoint/")
    config.graph_file = os.path.join(exp_dir, "checkpoint/","loss_training.npy")
    return config
