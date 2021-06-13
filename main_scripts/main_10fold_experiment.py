import os
from data_loader.data_generator import DataGenerator
from models.invariant_basic import invariant_basic
from trainers.trainer import Trainer
from Utils.config import process_config
from Utils.dirs import create_dirs
from Utils import doc_utils
from Utils.utils import get_args
import random
config = process_config('/Users/jiahe/PycharmProjects/gnn multiple inputs/configs/10fold_config.json')
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
import tensorflow.compat.v1 as tf
import numpy as np
from data_loader.data_helper import load_dataset_3s_val

tf.set_random_seed(1)
print("lr = {0}".format(config.learning_rate))
print("decay = {0}".format(config.decay_rate))
print(config.architecture)
# create the experiments dirs
create_dirs([config.summary_dir, config.checkpoint_dir])
doc_utils.doc_used_config(config)
for exp in range(1, 2):
    for fold in range(1, 11):
            print("Experiment num = {0}\nFold num = {1}".format(exp, fold))
            # create your data generator
            #config.num_fold = fold
            np.random.seed(fold)
            data = DataGenerator(config)
            gpuconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            gpuconfig.gpu_options.visible_device_list = config.gpus_list
            gpuconfig.gpu_options.allow_growth = True
            sess = tf.Session(config=gpuconfig)
            # create an instance of the model you want
            model = invariant_basic(config, data)
            # create trainer and pass all the previous components to it
            trainer = Trainer(sess, model, data, config)
            # here you train your model
            trainer.train()
            #  doc_utils.doc_results(acc, loss, exp, fold, config.summary_dir)
            sess.close()
            tf.reset_default_graph()

doc_utils.summary_10fold_results(config.summary_dir)




######### SYTHETIC DATA ##################
config = process_config('/Users/jiahe/PycharmProjects/gnn multiple inputs/configs/example.json')

tf.set_random_seed(1)
np.random.seed(1)
print("lr = {0}".format(config.learning_rate))
print("decay = {0}".format(config.decay_rate))
print(config.architecture)
# create the experiments dirs
create_dirs([config.summary_dir, config.checkpoint_dir])
doc_utils.doc_used_config(config)

for exp in range(1, 2):
    for fold in range(1, 11):
        print("Experiment num = {0}\nFold num = {1}".format(exp, fold))
        # create your data generator
        np.random.seed(fold)
        data = DataGenerator(config)
        random.seed(fold)
        data.val_graphs, data.val_labels, data.val_graphs3d = load_dataset_3s_val(config.dataset_name, config.upper)
        data.val_size = len(data.val_labels)
        gpuconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        gpuconfig.gpu_options.visible_device_list = config.gpus_list
        gpuconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=gpuconfig)
        # create an instance of the model you want
        model = invariant_basic(config, data)
        # create trainer and pass all the previous components to it
        trainer = Trainer(sess, model, data, config)
        # here you train your model
        trainer.train()
        #  doc_utils.doc_results(acc, loss, exp, fold, config.summary_dir)
        sess.close()
        tf.reset_default_graph()

doc_utils.summary_10fold_results(config.summary_dir)

