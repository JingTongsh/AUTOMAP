import os
import shutil
import tensorflow as tf

from data_loader.automap_data_generator import DataGenerator, ValDataGenerator
from models.automap_model import AUTOMAP_Basic_Model 
from trainers.automap_trainer import AUTOMAP_Trainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.args import get_args


def main():
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)
    
    # copy config file
    create_dirs([config.summary_dir, config.checkpoint_dir])
    shutil.copy(args.config, config.exp_dir)
    data = DataGenerator(config)
    valdata = ValDataGenerator(config)

    if config.resume ==0:
        model = AUTOMAP_Basic_Model(config)
    elif config.resume == 1:
        model = tf.keras.models.load_model(config.loadmodel_dir)
    model.summary()

    trainer = AUTOMAP_Trainer(model, data, valdata, config)
    trainer.train()


if __name__ == '__main__':
    main()
