import os
import sys
import copy

"""
How To:
Example for running from command line:
    python <path_to>/ProvablyPowerfulGraphNetworks/main_scripts/main_qm9_experiment.py --config=configs/qm9_config.json
"""
# Change working directory to project's main directory, and add it to path - for library and config usages
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)
import importlib
from data_loader.data_generator import DataGenerator, QM9_DataGenerator, QM9_DataGenerator_gnn3
from models.invariant_basic import invariant_basic, QM9_invariant_basic,QM9_invariant_basic2, QM9_invariant_basic_gnn3
from trainers.trainer import Trainer, QM9_Trainer, QM9_Trainer_gnn3
import trainers.trainer as trainers
importlib.reload(trainers)
from Utils.config import process_config
from Utils.dirs import create_dirs
from Utils import doc_utils
from Utils.utils import get_args
import tensorflow.compat.v1 as tf
import pandas as pd
tf.disable_eager_execution()
import random

def parametersearch():
    # capture the config path from the run arguments
    # then process the json configuration file
    config = process_config('/Users/jiahe/PycharmProjects/gnn multiple inputs/configs/parameter_search_config.json')
    data = QM9_DataGenerator(config)
    #train_labels_ori, val_labels_ori, test_labels_ori = copy.deepcopy(data.train_labels), copy.deepcopy(data.val_labels), copy.deepcopy(data.test_labels),
    data.train_labels, data.val_labels, data.test_labels = train_labels_ori[:, config.target_param].reshape(-1, 1),\
                 val_labels_ori[:, config.target_param].reshape(-1, 1),  test_labels_ori[:, config.target_param].reshape(-1, 1)

    base_summary_folder = config.summary_dir
    base_exp_name = config.exp_name

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    import numpy as np
    tf.set_random_seed(1)
   # for lr in [0.00008 * (2 ** i) for i in range(2, 8)]:
    param_grid = {
        'learning_rate': list(np.logspace(np.log10(0.00005), np.log10(0.1), base=10, num=1000)),
        'architecture1d': list(range(5, 500, 10)),
        'architecture2d': list(range(5, 500, 10)),
        'architecture3d': list(range(5, 500, 10)),
    }
    LR, A1, A2, A3 = [], [], [], []
    for expe in range(5) :
                hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
                lr, a1, a2, a3 = hyperparameters['learning_rate'], hyperparameters['architecture1d'], hyperparameters['architecture2d'], hyperparameters['architecture3d']
                LR.append(lr), A1.append(a1), A2.append(a2), A3.append(a3)
                config.exp_name = base_exp_name + "lr={0}_a1={1}_a2={2}=_a3={3}".format(lr, a1, a2, a3)
                curr_dir = os.path.join(base_summary_folder,
                             "lr={0}_a1={1}_a2={2}=_a3={3}".format(lr, a1, a2, a3))
                config.summary_dir = curr_dir
                 # create your data generator
                data.config.learning_rate = lr
                data.config.architecture1d = [a1]
                data.config.architecture2d = [a2]
                data.config.architecture3d = [a3]

                create_dirs([config.summary_dir, config.checkpoint_dir])
                doc_utils.doc_used_config(config)
                # create your data generator
                gpuconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
                gpuconfig.gpu_options.visible_device_list = config.gpus_list
                gpuconfig.gpu_options.allow_growth = True
                sess = tf.Session(config=gpuconfig)
                # create an instance of the model you want
                model = QM9_invariant_basic_gnn3(config, data)
                # create trainer and pass all the previous components to it
                trainer = QM9_Trainer_gnn3(sess, model, data, config)
                # here you train your model
                trainer.train()
                sess.close()
                tf.reset_default_graph()


import pandas as pd
def summary_10fold_results(summary_dir):
    df = pd.read_csv(summary_dir+"/per_epoch_stats.csv")
    acc = np.array(df["val_accuracy"])
    for i in range(len(acc)):
        acc[i] = float(''.join(list(acc[i])[1:-1]))
    print("Results")
    print("Mean MAR = {0}".format(np.mean(acc)))
 #   print("Mean std = {0}".format(np.std(acc)))

#for lr in [0.00008 * (2 ** i) for i in range(2, 8)]:
for lr in [0.00008*(2**i) for i in range(2,8)]:
                    dir = base_exp_name + "lr={0}".format(lr)
                    print('lr:' + str(lr))
                    summary_10fold_results(dir)


def main():
        # capture the config path from the run arguments
        # then process the json configuration file
        config = process_config('/Users/jiahe/PycharmProjects/gnn multiple inputs/configs/example.json')

        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
        import numpy as np
        tf.set_random_seed(1)
        print("lr = {0}".format(config.learning_rate))
        print("decay = {0}".format(config.decay_rate))
        if config.target_param is not False:  # (0 == False) while (0 is not False)
            print("target parameter: {0}".format(config.target_param))
        # create the experiments dirs
        create_dirs([config.summary_dir, config.checkpoint_dir])
        doc_utils.doc_used_config(config)
        data = QM9_DataGenerator(config)
        train_lables_ori, val_labels_ori, test_labels_ori = copy.deepcopy(data.train_labels), copy.deepcopy(data.val_labels), copy.deepcopy(data.test_labels),
        data.train_labels, data.val_labels, data.test_labels = train_lables_ori[:, config.target_param].reshape(-1, 1),\
                     val_labels_ori[:, config.target_param].reshape(-1, 1),  test_labels_ori[:, config.target_param].reshape(-1, 1)
        # create your data generator
        gpuconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        gpuconfig.gpu_options.visible_device_list = config.gpus_list
        gpuconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=gpuconfig)
        # create an instance of the model you want
        model = QM9_invariant_basic(config, data)
        # create trainer and pass all the previous components to it
        trainer = trainers.QM9_Trainer(sess, model, data, config)
        # here you train your model
        trainer.train()
        # test model, restore best model
        test_dists, test_loss, pred= trainer.test(load_best_model=True)
        sess.close()
        tf.reset_default_graph()

        doc_utils.summary_qm9_results(config.summary_dir, test_dists, test_loss, trainer.best_epoch)




############## gnn 3 ###############
    config = process_config('/Users/jiahe/PycharmProjects/gnn multiple inputs/configs/example.json')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    import numpy as np
    tf.set_random_seed(1)
    print("lr = {0}".format(config.learning_rate))
    print("decay = {0}".format(config.decay_rate))
    if config.target_param is not False:  # (0 == False) while (0 is not False)
        print("target parameter: {0}".format(config.target_param))
    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    doc_utils.doc_used_config(config)

    data = QM9_DataGenerator_gnn3(config)

    train_labels_ori, val_labels_ori, test_labels_ori = copy.deepcopy(data.train_labels), copy.deepcopy(data.val_labels), copy.deepcopy(data.test_labels),
    data.train_labels, data.val_labels, data.test_labels = train_labels_ori[:, config.target_param].reshape(-1, 1),\
                 val_labels_ori[:, config.target_param].reshape(-1, 1),  test_labels_ori[:, config.target_param].reshape(-1, 1)
    # create your data generator
    gpuconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    gpuconfig.gpu_options.visible_device_list = config.gpus_list
    gpuconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=gpuconfig)
    # create an instance of the model you want
    model = QM9_invariant_basic_gnn3(config, data)
    # create trainer and pass all the previous components to it
    trainer = trainers.QM9_Trainer_gnn3(sess, model, data, config)
    # here you train your model
    trainer.train()
    # test model, restore best model
    test_dists, test_loss, pred= trainer.test(load_best_model=True)
    sess.close()
    tf.reset_default_graph()

def parametersearch_gnn3():
    # capture the config path from the run arguments
    # then process the json configuration file
    config = process_config('/Users/jiahe/PycharmProjects/gnn multiple inputs/configs/parameter_search_config.json')
    data = QM9_DataGenerator_gnn3(config)
    #train_labels_ori, val_labels_ori, test_labels_ori = copy.deepcopy(data.train_labels), copy.deepcopy(data.val_labels), copy.deepcopy(data.test_labels),
    data.train_labels, data.val_labels, data.test_labels = train_labels_ori[:, config.target_param].reshape(-1, 1),\
                 val_labels_ori[:, config.target_param].reshape(-1, 1),  test_labels_ori[:, config.target_param].reshape(-1, 1)

    base_summary_folder = config.summary_dir
    base_exp_name = config.exp_name

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    import numpy as np
    tf.set_random_seed(1)
   # for lr in [0.00008 * (2 ** i) for i in range(2, 8)]:
    param_grid = {
        'learning_rate': list(np.logspace(np.log10(0.00005), np.log10(0.1), base=10, num=1000)),
        'architecture1d': [100],
        'architecture2d': [100],
        'architecture3d': [100],
    }
    for lr in [0.00008 * (2 ** i) for i in range(2, 8)]:
                config.exp_name = base_exp_name + "lr={0}".format(lr)
                curr_dir = os.path.join(base_summary_folder,
                             "lr={0}".format(lr))
                config.summary_dir = curr_dir
                 # create your data generator
                data.config.learning_rate = lr

                create_dirs([config.summary_dir, config.checkpoint_dir])
                doc_utils.doc_used_config(config)
                # create your data generator
                gpuconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
                gpuconfig.gpu_options.visible_device_list = config.gpus_list
                gpuconfig.gpu_options.allow_growth = True
                sess = tf.Session(config=gpuconfig)
                # create an instance of the model you want
                model = QM9_invariant_basic_gnn3(config, data)
                # create trainer and pass all the previous components to it
                trainer = trainers.QM9_Trainer_gnn3(sess, model, data, config)
                # here you train your model
                trainer.train()
                sess.close()
                tf.reset_default_graph()


    for lr in [0.00008*(2**i) for i in range(2,8)]:
                    dir = base_exp_name + "lr={0}".format(lr)
                    print('lr:' + str(lr))
                    summary_10fold_results(dir)
