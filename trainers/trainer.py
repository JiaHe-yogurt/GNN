from trainers.base_train import BaseTrain
import tensorflow.compat.v1 as tf
from tqdm import tqdm
import numpy as np
from Utils import doc_utils



class Trainer(BaseTrain):
    def __init__(self, sess, model, data, config):
        super(Trainer, self).__init__(sess, model, config, data)

        # load the model from the latest checkpoint if exist
        # self.model.load(self.sess)

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs, 1):
            # train epoch
            train_acc, train_loss = self.train_epoch(cur_epoch)
            self.sess.run(self.model.increment_cur_epoch_tensor)
            # validation step
            if self.config.val_exist:
                test_acc, test_loss, self.pred = self.test(cur_epoch)
                # document results
                doc_utils.write_to_file_doc(train_acc, train_loss, test_acc, test_loss, cur_epoch, self.config)
        if self.config.val_exist:
            # creates plots for accuracy and loss during training
            doc_utils.create_experiment_results_plot(self.config.exp_name, "accuracy", self.config.summary_dir)
            doc_utils.create_experiment_results_plot(self.config.exp_name, "loss", self.config.summary_dir, log=True)
        return self.test(cur_epoch)

    def train_epoch(self, num_epoch=None):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step

        Train one epoch
        :param epoch: cur epoch number
        :return accuracy and loss on train set
        """
        # initialize dataset
        self.data_loader.initialize(is_train=True)

        # initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_train), total=self.data_loader.num_iterations_train,
                  desc="epoch-{}-".format(num_epoch))

        total_loss = 0.
        total_correct = 0.

        # Iterate over batches
        for cur_it in tt:
            # One Train step on the current batch
            loss, correct = self.train_step()
            # update results from train_step func
            total_loss += loss
            total_correct += correct

        # save model
        if num_epoch % self.config.save_rate == 0:
            self.model.save(self.sess)

        loss_per_epoch = total_loss / self.data_loader.train_size
        acc_per_epoch = total_correct / self.data_loader.train_size
        print("""
        Epoch-{}  loss:{:.4f} -- acc:{:.4f}
                """.format(num_epoch, loss_per_epoch, acc_per_epoch))

        tt.close()
        return acc_per_epoch, loss_per_epoch

    def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - :return any accuracy and loss on current batch
       """

        # graphs,  labels = self.data_loader.next_batch()
        # _, loss, correct = self.sess.run([self.model.train_op, self.model.loss, self.model.correct_predictions],feed_dict={self.model.graphs: graphs, self.model.labels: labels, self.model.is_training: True})
        graphs, labels, graphs3d = self.data_loader.next_batch()
        _, loss, correct = self.sess.run([self.model.train_op, self.model.loss, self.model.correct_predictions],
                                         feed_dict={self.model.graphs1: graphs3d, self.model.labels: labels,
                                 self.model.graphs2: graphs, self.model.is_training: True})

        return loss, correct

    def test(self, epoch):
        # initialize dataset
        self.data_loader.initialize(is_train=False)

        # initialize tqdm
        tt = tqdm(range(self.data_loader.val_size), total=self.data_loader.val_size,
                  desc="Val-{}-".format(epoch))

        total_loss = 0.
        total_correct = 0.
        prediction = []
        # Iterate over batches
        for cur_it in tt:
            # One Train step on the current batch
            graph, label, graph3d = self.data_loader.next_batch()
            label = np.expand_dims(label, 0)
            pred, loss, correct = self.sess.run([self.model.pred, self.model.loss, self.model.correct_predictions],
                                                feed_dict={self.model.graphs1: graph3d, self.model.labels: label,
                                   self.model.graphs2: graph, self.model.is_training: False})
            # update metrics returned from train_step func
            prediction.append(pred)
            total_loss += loss
            total_correct += correct

        test_loss = total_loss / self.data_loader.val_size
        test_acc = total_correct / self.data_loader.val_size

        print("""
        Val-{}  loss:{:.4f} -- acc:{:.4f}
        """.format(epoch, test_loss, test_acc))

        tt.close()
        return test_acc, test_loss, np.array(prediction)


class QM9_Trainer(BaseTrain):
    def __init__(self, sess, model, data, config):
        super(QM9_Trainer, self).__init__(sess, model, config, data)
        self.is_QM9 = config.dataset_name == 'QM9'
        #self.request_from_model = self.model.sum_distances if self.is_QM9 else self.model.correct_predictions
        self.best_val_loss = np.inf
        self.best_epoch = -1

        # load the model from the latest checkpoint if exist
        # self.model.load(self.sess)

    def train(self):
        """
        Trains for the num of epochs in the config.
        :return:
        """
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs, 1):
            # train epoch
            train_acc, train_loss = self.train_epoch(cur_epoch)
            self.sess.run(self.model.increment_cur_epoch_tensor)
            # validation step
            if self.config.val_exist:
                val_acc, val_loss = self.validate(cur_epoch)
                # document results
                doc_utils.write_to_file_doc(train_acc, train_loss, val_acc, val_loss, cur_epoch, self.config)
        if self.config.val_exist:
            # creates plots for accuracy and loss during training
            if not self.is_QM9:
                doc_utils.create_experiment_results_plot(self.config.exp_name, "accuracy", self.config.summary_dir)
            doc_utils.create_experiment_results_plot(self.config.exp_name, "loss", self.config.summary_dir, log=True)

    def train_epoch(self, num_epoch=None):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step

        Train one epoch
        :param num_epoch: cur epoch number
        :return accuracy and loss on train set
        """
        # initialize dataset
        self.data_loader.initialize('train')

        # initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_train), total=self.data_loader.num_iterations_train,
                  desc="Epoch-{}-".format(num_epoch))

        total_loss = 0.
        total_correct_labels_or_distances = 0.

        # Iterate over batches
        for cur_it in tt:
            # One Train step on the current batch
            loss, correct_labels_or_distances = self.train_step()
            # update results from train_step func
            total_loss += loss
            total_correct_labels_or_distances += correct_labels_or_distances

        tt.close()

        loss_per_epoch = total_loss/self.data_loader.train_size
        dist_per_epoch = (total_correct_labels_or_distances * self.data_loader.labels_std[self.model.config.target_param])/self.data_loader.train_size
        print("\t\tEpoch-{}  loss:{} -- mean_distances:\n{}\n".format(num_epoch, loss_per_epoch, dist_per_epoch))
        return dist_per_epoch, loss_per_epoch

    def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - :return any accuracy and loss on current batch
       """

        graphs1d, graphs2d, graphs3d, labels  = self.data_loader.next_batch()
        pred, _, loss, correct = self.sess.run([self.model.pred,self.model.train_op, self.model.loss, self.model.correct_predictions],
                                         feed_dict={self.model.graphs1: graphs1d,self.model.graphs2: graphs2d, self.model.graphs3: graphs3d, self.model.labels: labels,
                                                    self.model.is_training: True})
        return loss, correct

    def validate(self, epoch):
        """
        Perform forward pass on the model with the validation set
        :param epoch: Epoch number
        :return: (val_acc, val_loss) for benchmark graphs, (val_dists, val_loss) for QM9
        """
        # initialize dataset
        self.data_loader.initialize('val')

        # initialize tqdm
        # tt = tqdm(range(self.data_loader.num_iterations_val), total=self.data_loader.num_iterations_val,
        #           desc="Val-{}-".format(epoch))

        total_loss = 0.
        total_correct_or_dist = 0.

        # Iterate over batches
        # for cur_it in tt:
        for cur_it in range(self.data_loader.num_iterations_val):
            # One Train step on the current batch
            graphs1d, graphs2d, graphs3d, label = self.data_loader.next_batch()
            label = label[:, self.model.config.target_param].reshape(-1, 1)

            loss, correct_or_dist = self.sess.run([self.model.loss, self.model.correct_predictions],
                                             feed_dict={self.model.graphs1: graphs1d, self.model.graphs2: graphs2d, self.model.graphs3: graphs3d, self.model.labels: label, self.model.is_training: False})
            # update metrics returned from train_step func
            total_loss += loss
            total_correct_or_dist += correct_or_dist

        # tt.close()

        val_loss = total_loss/self.data_loader.val_size
        val_dists = (total_correct_or_dist * self.data_loader.labels_std[self.model.config.target_param])/self.data_loader.val_size
        print("\t\tVal-{}  loss:{} -- mean_distances:\n{}\n".format(epoch, val_loss, val_dists))

            # save best model by validation loss to be used for test set
        if self.config.test_exist == True:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print("New best validation score achieved.")
                self.model.save(self.sess)
                self.best_epoch = epoch
        return val_dists, val_loss


    def test(self, load_best_model=False):
        """
        Perform forward pass on the model for the test set
        :param load_best_model: Boolean. True for loading the best model saved, based on validation loss
        :return: (test_dists, test_loss)
        """
        # load best saved model
        if load_best_model:
            self.model.load(self.sess)

        # initialize dataset
        self.data_loader.initialize('test')

        # initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_test), total=self.data_loader.num_iterations_test,
                  desc="Test-{}-".format(self.best_epoch))

        total_loss = 0.
        total_dists = 0.
        prediction = []

        # Iterate over batches
        for cur_it in tt:
            # One Train step on the current batch
            graphs1d, graphs2d, graphs3d, label = self.data_loader.next_batch()
            label = label[:, self.model.config.target_param].reshape(-1, 1)

            pred, loss, dists = self.sess.run([self.model.pred, self.model.loss, self.model.correct_predictions],
                                        feed_dict={self.model.graphs1: graphs1d, self.model.graphs2: graphs2d, self.model.graphs3: graphs3d, self.model.labels: label, self.model.is_training: False})
            # update metrics returned from train_step func
            total_loss += loss
            total_dists += dists
            prediction.append(pred)

        test_loss = total_loss/self.data_loader.test_size
        test_dists = (total_dists * self.data_loader.labels_std[self.model.config.target_param]) / self.data_loader.test_size
        print("\t\tTest-{}  loss:{} -- mean_distances: {}".format(self.best_epoch, test_loss, test_dists))

        tt.close()

        return test_dists, test_loss, np.array(prediction)


class QM9_Trainer_gnn3(BaseTrain):
    def __init__(self, sess, model, data, config):
        super(QM9_Trainer_gnn3, self).__init__(sess, model, config, data)
        self.is_QM9 = config.dataset_name == 'QM9'
        #self.request_from_model = self.model.sum_distances if self.is_QM9 else self.model.correct_predictions
        self.best_val_loss = np.inf
        self.best_epoch = -1

        # load the model from the latest checkpoint if exist
        # self.model.load(self.sess)

    def train(self):
        """
        Trains for the num of epochs in the config.
        :return:
        """
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs, 1):
            # train epoch
            train_acc, train_loss = self.train_epoch(cur_epoch)
            self.sess.run(self.model.increment_cur_epoch_tensor)
            # validation step
            if self.config.val_exist:
                val_acc, val_loss = self.validate(cur_epoch)
                # document results
                doc_utils.write_to_file_doc(train_acc, train_loss, val_acc, val_loss, cur_epoch, self.config)
        if self.config.val_exist:
            # creates plots for accuracy and loss during training
            if not self.is_QM9:
                doc_utils.create_experiment_results_plot(self.config.exp_name, "accuracy", self.config.summary_dir)
            doc_utils.create_experiment_results_plot(self.config.exp_name, "loss", self.config.summary_dir, log=True)

    def train_epoch(self, num_epoch=None):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step

        Train one epoch
        :param num_epoch: cur epoch number
        :return accuracy and loss on train set
        """
        # initialize dataset
        self.data_loader.initialize('train')

        # initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_train), total=self.data_loader.num_iterations_train,
                  desc="Epoch-{}-".format(num_epoch))

        total_loss = 0.
        total_correct_labels_or_distances = 0.

        # Iterate over batches
        for cur_it in tt:
            # One Train step on the current batch
            loss, correct_labels_or_distances = self.train_step()
            # update results from train_step func
            total_loss += loss
            total_correct_labels_or_distances += correct_labels_or_distances

        tt.close()

        loss_per_epoch = total_loss/self.data_loader.train_size
        dist_per_epoch = (total_correct_labels_or_distances * self.data_loader.labels_std[self.model.config.target_param])/self.data_loader.train_size
        print("\t\tEpoch-{}  loss:{} -- mean_distances:\n{}\n".format(num_epoch, loss_per_epoch, dist_per_epoch))
        return dist_per_epoch, loss_per_epoch

    def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - :return any accuracy and loss on current batch
       """

        graphs2d, labels, graphs3d  = self.data_loader.next_batch()
        pred, _, loss, correct = self.sess.run([self.model.pred,self.model.train_op, self.model.loss, self.model.correct_predictions],
                                         feed_dict={self.model.graphs2: graphs2d, self.model.graphs1: graphs3d, self.model.labels: labels,
                                                    self.model.is_training: True})
        return loss, correct

    def validate(self, epoch):
        """
        Perform forward pass on the model with the validation set
        :param epoch: Epoch number
        :return: (val_acc, val_loss) for benchmark graphs, (val_dists, val_loss) for QM9
        """
        # initialize dataset
        self.data_loader.initialize('val')

        # initialize tqdm
        # tt = tqdm(range(self.data_loader.num_iterations_val), total=self.data_loader.num_iterations_val,
        #           desc="Val-{}-".format(epoch))

        total_loss = 0.
        total_correct_or_dist = 0.

        # Iterate over batches
        # for cur_it in tt:
        for cur_it in range(self.data_loader.num_iterations_val):
            # One Train step on the current batch
            graphs2d, label, graphs3d = self.data_loader.next_batch()
            label = label[:, self.model.config.target_param].reshape(-1, 1)

            loss, correct_or_dist = self.sess.run([self.model.loss, self.model.correct_predictions],
                                             feed_dict={self.model.graphs2: graphs2d, self.model.graphs1: graphs3d, self.model.labels: label, self.model.is_training: False})
            # update metrics returned from train_step func
            total_loss += loss
            total_correct_or_dist += correct_or_dist

        # tt.close()

        val_loss = total_loss/self.data_loader.val_size
        val_dists = (total_correct_or_dist * self.data_loader.labels_std[self.model.config.target_param])/self.data_loader.val_size
        print("\t\tVal-{}  loss:{} -- mean_distances:\n{}\n".format(epoch, val_loss, val_dists))

            # save best model by validation loss to be used for test set
        if self.config.test_exist == True:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print("New best validation score achieved.")
                self.model.save(self.sess)
                self.best_epoch = epoch
        return val_dists, val_loss


    def test(self, load_best_model=False):
        """
        Perform forward pass on the model for the test set
        :param load_best_model: Boolean. True for loading the best model saved, based on validation loss
        :return: (test_dists, test_loss)
        """
        # load best saved model
        if load_best_model:
            self.model.load(self.sess)

        # initialize dataset
        self.data_loader.initialize('test')

        # initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_test), total=self.data_loader.num_iterations_test,
                  desc="Test-{}-".format(self.best_epoch))

        total_loss = 0.
        total_dists = 0.
        prediction = []

        # Iterate over batches
        for cur_it in tt:
            # One Train step on the current batch
            graphs2d, label , graphs3d= self.data_loader.next_batch()
            label = label[:, self.model.config.target_param].reshape(-1, 1)

            pred, loss, dists = self.sess.run([self.model.pred, self.model.loss, self.model.correct_predictions],
                                        feed_dict={ self.model.graphs2: graphs2d, self.model.graphs1: graphs3d, self.model.labels: label, self.model.is_training: False})
            # update metrics returned from train_step func
            total_loss += loss
            total_dists += dists
            prediction.append(pred)

        test_loss = total_loss/self.data_loader.test_size
        test_dists = (total_dists * self.data_loader.labels_std[self.model.config.target_param]) / self.data_loader.test_size
        print("\t\tTest-{}  loss:{} -- mean_distances: {}".format(self.best_epoch, test_loss, test_dists))

        tt.close()

        return test_dists, test_loss, np.array(prediction)
