import numpy as np
import data_loader.data_helper as helper
import Utils.config

class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        self.batch_size = self.config.batch_size
        self.load_data()

    # load the specified dataset in the config to the data_generator instance
    def load_data(self):
        if self.config.network == "gnn3":
           graphs, labels, graphs3d = helper.gnn3(self.config.dataset_name, self.config.target_shape)
        elif self.config.network =="gnn4":
           graphs, labels, graphs3d = helper.gnn4(self.config.dataset_name, self.config.target_shape)
      #  graphs, labels, graphs3d = helper.load_dataset3s(self.config.dataset_name)
        # if no fold specify creates random split to train and validation
        if self.config.num_fold is None:
            graphs, labels, graphs3d = helper.shuffle(graphs, labels, graphs3d)
            idx = len(graphs) // 10
            self.train_graphs, self.train_labels,  self.train_graphs3d, self.val_graphs, self.val_labels, self.val_graphs3d  = graphs[idx:], labels[idx:], graphs3d[idx:], graphs[:idx], labels[:idx], graphs3d[:idx]
        elif self.config.num_fold == 0:
            train_idx, test_idx = helper.get_parameter_split(self.config.dataset_name)
            self.train_graphs, self.train_labels,  self.train_graphs3d, self.val_graphs, self.val_labels , self.val_graphs3d = graphs[train_idx], labels[
                train_idx],  graphs3d[train_idx], graphs[test_idx], labels[test_idx], graphs3d[test_idx]
        else:
            train_idx, test_idx = helper.get_train_val_indexes(self.config.num_fold, self.config.dataset_name)
            self.train_graphs, self.train_labels,  self.train_graphs3d, self.val_graphs, self.val_labels , self.val_graphs3d = graphs[train_idx], labels[
                train_idx],  graphs3d[train_idx], graphs[test_idx], labels[test_idx], graphs3d[test_idx]
        # change validation graphs to the right shape
        self.val_graphs = [np.expand_dims(g, 0) for g in self.val_graphs]
        self.val_graphs3d = [np.expand_dims(g, 0) for g in self.val_graphs3d]
        self.train_size = len(self.train_graphs)
        self.val_size = len(self.val_graphs)

    def next_batch(self):
        return next(self.iter)

    # initialize an iterator from the data for one training epoch
    def initialize(self, is_train):
        if is_train:
            self.reshuffle_data()
        else:
            self.iter = zip(self.val_graphs, self.val_labels, self.val_graphs3d)

    # resuffle data iterator between epochs
    def reshuffle_data(self):
        graphs, labels, graphs3d = helper.group_same_size(self.train_graphs, self.train_labels, self.train_graphs3d)
        graphs, labels, graphs3d = helper.shuffle_same_size(graphs, labels, graphs3d)
        graphs, labels, graphs3d = helper.split_to_batches(graphs, labels, graphs3d, self.batch_size)
        self.num_iterations_train = len(graphs)
        graphs, labels, graphs3d = helper.shuffle(graphs, labels, graphs3d)
        self.iter = zip(graphs, labels, graphs3d)


class QM9_DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        self.batch_size = self.config.batch_size
        self.load_data()

    # load the specified dataset in the config to the data_generator instance
    def load_data(self):
        self.load_qm9_data()
        self.split_val_test_to_batches()

    # load QM9 data set
    def load_qm9_data(self):
        train_graphs1d, train_graphs2d, train_graphs3d, train_labels, val_graphs1d, val_graphs2d, val_graphs3d,\
        val_labels, test_graphs1d, test_graphs2d, test_graphs3d, test_labels= helper.load_qm9(self.config.target_param, self.config.target_shape)

        # preprocess all labels by train set mean and std
        train_labels_mean = train_labels.mean(axis=0)
        train_labels_std = train_labels.std(axis=0)
        train_labels = (train_labels - train_labels_mean) / train_labels_std
        val_labels = (val_labels - train_labels_mean) / train_labels_std
        test_labels = (test_labels - train_labels_mean) / train_labels_std

        self.train_graphs1d,self.train_graphs2d,self.train_graphs3d, self.train_labels = train_graphs1d, train_graphs2d, train_graphs3d, train_labels
        self.val_graphs1d, self.val_graphs2d,self.val_graphs3d, self.val_labels = val_graphs1d, val_graphs2d, val_graphs3d, val_labels
        self.test_graphs1d, self.test_graphs2d,self.test_graphs3d, self.test_labels = test_graphs1d, test_graphs2d, test_graphs3d, test_labels

        self.train_size = len(self.train_graphs1d)
        self.val_size = len(self.val_graphs1d)
        self.test_size = len(self.test_graphs1d)
        self.labels_std = train_labels_std  # Needed for postprocess, multiply mean abs distance by this std


    def next_batch(self):
        return next(self.iter)

    # initialize an iterator from the data for one training epoch
    def initialize(self, what_set):
        if what_set == 'train':
            self.reshuffle_data()
        elif what_set == 'val' or what_set == 'validation':
            self.iter = zip(self.val_graphs1d_batches,self.val_graphs2d_batches,self.val_graphs3d_batches, self.val_labels_batches)
        elif what_set == 'test':
            self.iter = zip(self.test_graphs1d_batches,self.test_graphs2d_batches,self.test_graphs3d_batches, self.test_labels_batches)
        else:
            raise ValueError("what_set should be either 'train', 'val' or 'test'")

    def reshuffle_data(self):
        """
        Reshuffle train data between epochs
        """
        graphs1d, graphs2d,graphs3d, labels = helper.QM9_group_same_size(self.train_graphs1d,self.train_graphs2d,self.train_graphs3d, self.train_labels)
        graphs1d, graphs2d,graphs3d, labels = helper.QM9_shuffle_same_size(graphs1d, graphs2d,graphs3d, labels)
        graphs1d, graphs2d,graphs3d, labels = helper.QM9_split_to_batches(graphs1d, graphs2d,graphs3d, labels, self.batch_size)
        self.num_iterations_train = len(graphs1d)
        graphs1d, graphs2d,graphs3d, labels = helper.QM9_shuffle(graphs1d, graphs2d,graphs3d, labels)
        self.iter = zip(graphs1d, graphs2d,graphs3d, labels)


    def split_val_test_to_batches(self):
        # Split the val and test sets to batchs, no shuffling is needed
        graphs1d, graphs2d, graphs3d, labels = helper.QM9_group_same_size(self.val_graphs1d,self.val_graphs2d,self.val_graphs3d, self.val_labels)
        graphs1d, graphs2d, graphs3d, labels = helper.QM9_split_to_batches(graphs1d, graphs2d, graphs3d, labels, self.batch_size)
        self.num_iterations_val = len(graphs1d)
        self.val_graphs1d_batches,self.val_graphs2d_batches, self.val_graphs3d_batches, self.val_labels_batches = graphs1d, graphs2d, graphs3d, labels

        graphs1d, graphs2d, graphs3d, labels = helper.QM9_group_same_size(self.test_graphs1d,self.test_graphs2d,self.test_graphs3d, self.test_labels)
        graphs1d, graphs2d, graphs3d, labels = helper.QM9_split_to_batches(graphs1d, graphs2d, graphs3d, labels, self.batch_size)
        self.num_iterations_test = len(graphs1d)
        self.test_graphs1d_batches,self.test_graphs2d_batches, self.test_graphs3d_batches, self.test_labels_batches = graphs1d, graphs2d, graphs3d, labels


class QM9_DataGenerator_gnn3:
    def __init__(self, config):
        self.config = config
        # load data here
        self.batch_size = self.config.batch_size
        self.load_data()

    # load the specified dataset in the config to the data_generator instance
    def load_data(self):
        self.load_qm9_data()
        self.split_val_test_to_batches()

    # load QM9 data set
    def load_qm9_data(self):
        train_graphs, train_labels, train_graphs3d, val_graphs, val_labels, val_graphs3d, \
        test_graphs, test_labels, test_graphs3d = helper.load_qm9_gnn3(self.config.target_param, self.config.target_shape)

        # preprocess all labels by train set mean and std
        train_labels_mean = train_labels.mean(axis=0)
        train_labels_std = train_labels.std(axis=0)
        train_labels = (train_labels - train_labels_mean) / train_labels_std
        val_labels = (val_labels - train_labels_mean) / train_labels_std
        test_labels = (test_labels - train_labels_mean) / train_labels_std

        self.train_graphs, self.train_graphs3d, self.train_labels = train_graphs, train_graphs3d, train_labels
        self.val_graphs, self.val_graphs3d, self.val_labels = val_graphs, val_graphs3d, val_labels
        self.test_graphs, self.test_graphs3d, self.test_labels = test_graphs, test_graphs3d, test_labels

        self.train_size = len(self.train_graphs)
        self.val_size = len(self.val_graphs)
        self.test_size = len(self.test_graphs)
        self.labels_std = train_labels_std  # Needed for postprocess, multiply mean abs distance by this std


    def next_batch(self):
        return next(self.iter)

    # initialize an iterator from the data for one training epoch
    def initialize(self, what_set):
        if what_set == 'train':
            self.reshuffle_data()
        elif what_set == 'val' or what_set == 'validation':
            self.iter = zip(self.val_graphs_batches, self.val_labels_batches, self.val_graphs3d_batches)
        elif what_set == 'test':
            self.iter = zip(self.test_graphs_batches, self.test_labels_batches, self.test_graphs3d_batches)
        else:
            raise ValueError("what_set should be either 'train', 'val' or 'test'")

    def reshuffle_data(self):
        """
        Reshuffle train data between epochs
        """
        graphs, labels,graphs3d = helper.group_same_size(self.train_graphs, self.train_labels, self.train_graphs3d)
        graphs, labels,graphs3d = helper.shuffle_same_size( graphs, labels, graphs3d)
        graphs, labels,graphs3d = helper.split_to_batches(graphs, labels, graphs3d, self.batch_size)
        self.num_iterations_train = len(graphs)
        graphs, labels,graphs3d = helper.shuffle(graphs, labels, graphs3d)
        self.iter = zip( graphs,labels,graphs3d)


    def split_val_test_to_batches(self):
        # Split the val and test sets to batchs, no shuffling is needed
        graphs, labels,graphs3d = helper.group_same_size(self.val_graphs, self.val_labels, self.val_graphs3d)
        graphs, labels,graphs3d = helper.split_to_batches( graphs, labels, graphs3d,  self.batch_size)
        self.num_iterations_val = len(graphs)
        self.val_graphs_batches, self.val_graphs3d_batches, self.val_labels_batches = graphs, graphs3d, labels

        graphs, labels, graphs3d = helper.group_same_size(self.test_graphs, self.test_labels, self.test_graphs3d)
        graphs, labels, graphs3d = helper.split_to_batches( graphs, labels, graphs3d,  self.batch_size)
        self.num_iterations_test = len(graphs)
        self.test_graphs_batches, self.test_graphs3d_batches, self.test_labels_batches = graphs, graphs3d, labels











class DataGenerator_quick:
    def __init__(self, config,Graphs,Labels):
        self.config = config
        # load data here
        self.batch_size = self.config.batch_size
        self.load_data(Graphs,Labels)

    # load the specified dataset in the config to the data_generator instance
    def load_data(self, Graphs, Labels):
        if self.config.num_fold is None:
            graphs, labels = helper.shuffle(Graphs, Labels)
            idx = len(graphs) // 10
            self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = graphs[idx:], labels[idx:], graphs[:idx], labels[:idx]
        elif self.config.num_fold == 0:
            train_idx, test_idx = helper.get_parameter_split(self.config.dataset_name)
            self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = graphs[train_idx], labels[
                train_idx], graphs[test_idx], labels[test_idx]
        else:
            train_idx, test_idx = helper.get_train_val_indexes(self.config.num_fold, self.config.dataset_name)
            self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = graphs[train_idx], labels[train_idx], graphs[test_idx], labels[
                test_idx]
        # change validation graphs to the right shape
        self.val_graphs = [np.expand_dims(g, 0) for g in self.val_graphs]
        self.train_size = len(self.train_graphs)
        self.val_size = len(self.val_graphs)

    def next_batch(self):
        return next(self.iter)

    # initialize an iterator from the data for one training epoch
    def initialize(self, is_train):
        if is_train:
            self.reshuffle_data()
        else:
            self.iter = zip(self.val_graphs, self.val_labels)

    # resuffle data iterator between epochs
    def reshuffle_data(self):
        graphs, labels = helper.group_same_size(self.train_graphs, self.train_labels)
        graphs, labels = helper.shuffle_same_size(graphs, labels)
        graphs, labels = helper.split_to_batches(graphs, labels, self.batch_size)
        self.num_iterations_train = len(graphs)
        graphs, labels = helper.shuffle(graphs, labels)
        self.iter = zip(graphs, labels)


if __name__ == '__main__':
    config = utils.config.process_config('../configs/example.json')
    data = DataGenerator(config)
    data.initialize(True)
