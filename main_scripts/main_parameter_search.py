import os
from data_loader.data_generator import DataGenerator
from models.invariant_basic import invariant_basic
from trainers.trainer import Trainer
from Utils.config import process_config
from Utils.dirs import create_dirs
from Utils import doc_utils
from Utils.utils import get_args
from data_loader import data_helper as helper
# capture the config path from the run arguments
# then process the json configuration file
config = process_config('/Users/jiahe/PycharmProjects/gnn multiple inputs/configs/parameter_search_config.json')
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
import tensorflow.compat.v1 as tf
import numpy as np

tf.set_random_seed(1)
base_summary_folder = config.summary_dir
base_exp_name = config.exp_name
# create the experiments dirs
create_dirs([config.summary_dir, config.checkpoint_dir])
data = DataGenerator(config)

for lr in [0.00008*(2**i) for i in range(2,8)]:
    for a1d in [[5],[10]]:
        for a3d in [[5], [10],[15]]:
          for fully in [[50,50],[20,20]]:

            config.learning_rate = lr
            config.architecture2d = a1d
            config.architecture = a3d
            config.fc = fully
            config.exp_name = base_exp_name + " lr={0}_a2d={1}_a3d = {2}_fc = {3}".format(lr, a1d,a3d,fully)
            curr_dir = os.path.join(base_summary_folder, "lr={0}_a2d={1}_a3d = {2}_fc = {3}".format(lr, a1d, a3d, fully))
            config.summary_dir = curr_dir
            create_dirs([curr_dir])
            # create your data generator
            data.config.learning_rate=lr
            data.config.architecture2d = a1d
            data.config.architecture3d = a3d
            data.config.fc = fully
            gpuconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            gpuconfig.gpu_options.visible_device_list = config.gpus_list
            gpuconfig.gpu_options.allow_growth = True
            sess = tf.Session(config=gpuconfig)
            # create an instance of the model you want
            model = invariant_basic(config, data)
            # create trainer and pass all the previous components to it
            trainer = Trainer(sess, model, data, config)
            # here you train your model
            acc, loss, _ = trainer.train()
            sess.close()
            tf.reset_default_graph()


import pandas as pd
def summary_10fold_results(summary_dir):
    df = pd.read_csv(summary_dir+"/per_epoch_stats.csv")
    acc = np.array(df["val_accuracy"])
    print("Results")
    print("Mean Accuracy = {0}".format(np.mean(acc)))
 #   print("Mean std = {0}".format(np.std(acc)))
    return np.mean(acc)

