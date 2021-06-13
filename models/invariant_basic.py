from models.base_model import BaseModel
import layers.equivariant_linear as eq
import layers.layers as layers
import tensorflow.compat.v1 as tf


class invariant_basic(BaseModel):
    def __init__(self, config, data):
        super(invariant_basic, self).__init__(config)
        self.data = data
        self.build_model()
        self.init_saver()

    def build_model(self):
        # here you build the tensorflow graph of any model you want and define the loss.
        self.is_training = tf.placeholder(tf.bool)

        self.labels = tf.placeholder(tf.int32, shape=[None])
        ## for A^3
        if self.config.input_order == 3:
            self.graphs1 = tf.placeholder(tf.float32, shape=[None, self.data.train_graphs3d[0].shape[0], None, None, None])
            net3d = eq.equi_3_to_1('tri_equi0', self.data.train_graphs3d[0].shape[0], self.config.architecture3d[0], self.graphs1)
            net3d = tf.nn.relu(net3d, name='rel0')
            if self.config.network == 'gnn3':
                self.graphs2 = tf.placeholder(tf.float32, shape=[None, self.data.train_graphs[0].shape[0], None, None])
                net2d = eq.equi_2_to_2('2d_equ0', self.data.train_graphs[0].shape[0], self.config.architecture2d[0],
                                       self.graphs2)
                net2d = tf.nn.relu(net2d, name='rel0')
                for layer in range(1, len(self.config.architecture2d)):  # architecture is # of features of each layers
                    net2d = eq.equi_2_to_2('2d_equ%d' % layer, self.config.architecture2d[layer - 1],
                                           self.config.architecture2d[layer], net2d)
                    net2d = tf.nn.relu(net2d, name='2d_rel%d' % layer)
                net2d = layers.diag_offdiag_maxpool(net2d)  # invariant max layer according to the invariant basis
                net = tf.concat([net2d, net3d], axis=1)

            elif self.config.network == 'gnn4':
                self.graphs2 = tf.placeholder(tf.float32, shape=[None, self.data.train_graphs[0].shape[0], None])
                net1d = eq.equi_1_to_1('2d_equ0', self.data.train_graphs[0].shape[0], self.config.architecture1d[0],
                                       self.graphs2)
                net1d = tf.nn.relu(net1d, name='rel0')
                for layer in range(1, len(self.config.architecture2d)):  # architecture is # of features of each layers
                    net1d = eq.equi_1_to_1('2d_equ%d' % layer, self.config.architecture1d[layer - 1],
                                           self.config.architecture2d[layer], net1d)
                    net1d = tf.nn.relu(net1d, name='2d_rel%d' % layer)
                net1d = tf.reduce_sum(net1d,axis=2)
                net = tf.concat([net1d, net3d], axis=1)

        ## for A^4
        elif self.config.input_order == 4:
            self.graphs1 = tf.placeholder(tf.float32, shape=[None, 1, None, None, None, None])
            net4d = eq.equi_4_to_1('four_equi0', 1, self.config.architecture[0], self.graphs1)
            net4d = tf.nn.relu(net4d, name='rel0')
            self.graphs2 = tf.placeholder(tf.float32, shape=[None, self.data.train_graphs[0].shape[0], None, None])
            net2d = eq.equi_2_to_2('2d_equ0', self.data.train_graphs[0].shape[0], self.config.architecture2d[0],
                                   self.graphs2)
            net2d = tf.nn.relu(net2d, name='rel0')
            for layer in range(1, len(self.config.architecture2d)):  # architecture is # of features of each layers
                net2d = eq.equi_2_to_2('2d_equ%d' % layer, self.config.architecture2d[layer - 1],
                                       self.config.architecture2d[layer], net2d)
                net2d = tf.nn.relu(net2d, name='2d_rel%d' % layer)
            net2d = layers.diag_offdiag_maxpool(net2d)  # invariant max layer according to the invariant basis

            net = tf.concat([net2d, net4d], axis=1)


        net = layers.fully_connected(net, self.config.fc[0], "full1")
        net = layers.fully_connected(net, self.config.fc[1], "full2")
        net = layers.fully_connected(net, self.config.num_classes, "full4", activation_fn=None)  # original classification
      #  net = tf.reshape(layers.fully_connected(net, 1, "fully3", activation_fn=None),(-1,))      # regression
        # define loss function
        with tf.name_scope("loss"):
            self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=net))
            self.correct_predictions = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(net, 1, output_type=tf.int32), self.labels), tf.int32))
            self.pred = tf.argmax(net, 1, output_type=tf.int32)
       #       self.loss = tf.reduce_sum(tf.losses.mean_squared_error(labels=self.labels, predictions=net))        # regression
       #       self.correct_predictions = tf.reduce_sum(tf.losses.mean_squared_error(labels=self.labels, predictions=net))        # regression
       #       self.pred = net
        # get learning rate with decay every 20 epochs
        learning_rate = self.get_learning_rate(self.global_step_tensor, self.data.train_size * 20)

        # choose optimizer
        if self.config.optimizer == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=self.config.momentum)
        elif self.config.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate)

        # define train step
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def get_learning_rate(self, global_step, decay_step):
        """
        helper method to fit learning rat
        :param global_step: current index into dataset, int
        :param decay_step: decay step, float
        :return: output: N x S x m x m tensor
        """
        learning_rate = tf.train.exponential_decay(
            self.config.learning_rate,  # Base learning rate.
            global_step * self.config.batch_size,
            decay_step,
            self.config.decay_rate,  # Decay rate.
            staircase=True)
        learning_rate = tf.maximum(learning_rate, 0.00001)
        return learning_rate


class QM9_invariant_basic(BaseModel):
    def __init__(self, config, data):
        super(QM9_invariant_basic, self).__init__(config)
        self.data = data
        self.build_model()
        self.init_saver()

    def build_model(self):
        # here you build the tensorflow graph of any model you want and define the loss.
        self.is_training = tf.placeholder(tf.bool)

        self.labels = tf.placeholder(tf.int32, shape=[None,1])
        ## for A^3
        self.graphs3 = tf.placeholder(tf.float32, shape=[None, self.data.train_graphs3d[0].shape[0], None, None, None])
        net3d = eq.equi_3_to_1('tri_equi0', self.data.train_graphs3d[0].shape[0], self.config.architecture3d[0], self.graphs3)
        net3d = tf.nn.relu(net3d, name='rel0')
        self.graphs2 = tf.placeholder(tf.float32, shape=[None, self.data.train_graphs2d[0].shape[0], None, None])
        net2d = eq.equi_2_to_2('2d_equ0', self.data.train_graphs2d[0].shape[0], self.config.architecture2d[0],
                                       self.graphs2)
        net2d = tf.nn.relu(net2d, name='rel0')
        for layer in range(1, len(self.config.architecture2d)):  # architecture is # of features of each layers
              net2d = eq.equi_2_to_2('2d_equ%d' % layer, self.config.architecture2d[layer - 1],
                                           self.config.architecture2d[layer], net2d)
              net2d = tf.nn.relu(net2d, name='2d_rel%d' % layer)
        net2d = layers.diag_offdiag_maxpool(net2d)  # invariant max layer according to the invariant basis

        self.graphs1 = tf.placeholder(tf.float32, shape=[None, self.data.train_graphs1d[0].shape[0], None])
        net1d = eq.equi_1_to_1('1d_equ0', self.data.train_graphs1d[0].shape[0], self.config.architecture1d[0],
                                       self.graphs1)
        net1d = tf.nn.relu(net1d, name='rel1')
        for layer in range(1, len(self.config.architecture2d)):  # architecture is # of features of each layers
            net1d = eq.equi_1_to_1('1d_equ%d' % layer, self.config.architecture1d[layer - 1],
                                           self.config.architecture2d[layer], net1d)
            net1d = tf.nn.relu(net1d, name='1d_rel%d' % layer)
        net1d = tf.reduce_sum(net1d,axis=2)
        net = tf.concat([net1d, net2d, net3d], axis=1)
        net = layers.fully_connected(net, self.config.fc[0], "full1")
        net = layers.fully_connected(net, self.config.fc[1], "full2")
        net = layers.fully_connected(net, 1, "fully3", activation_fn=None)      # regression
        # define loss function
        with tf.name_scope("loss"):
            distances = tf.losses.absolute_difference(labels=self.labels, predictions=net,
                                                      reduction=tf.losses.Reduction.NONE)

            self.loss = tf.reduce_sum(distances, axis=0)
            self.correct_predictions = tf.reduce_sum(distances, axis=0)
            self.pred = net
        # get learning rate with decay every 20 epochs
        learning_rate = self.get_learning_rate(self.global_step_tensor, self.data.train_size * self.config.decay_epoch)

        # choose optimizer
        if self.config.optimizer == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=self.config.momentum)
        elif self.config.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate)

        # define train step
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def get_learning_rate(self, global_step, decay_step):
        """
        helper method to fit learning rat
        :param global_step: current index into dataset, int
        :param decay_step: decay step, float
        :return: output: N x S x m x m tensor
        """
        learning_rate = tf.train.exponential_decay(
            self.config.learning_rate,  # Base learning rate.
            global_step * self.config.batch_size,
            decay_step,
            self.config.decay_rate,  # Decay rate.
            staircase=True)
        learning_rate = tf.maximum(learning_rate, 0.00001)
        return learning_rate


class QM9_invariant_basic2(BaseModel):
    def __init__(self, config, data):
        super(QM9_invariant_basic2, self).__init__(config)
        self.data = data
        self.build_model()
        self.init_saver()

    def build_model(self):
        # here you build the tensorflow graph of any model you want and define the loss.
        self.is_training = tf.placeholder(tf.bool)

        self.labels = tf.placeholder(tf.int32, shape=[None,1])
        ## for A^3
        self.graphs1 = tf.placeholder(tf.float32, shape=[None, self.data.train_graphs1d[0].shape[0], None])

        self.graphs3 = tf.placeholder(tf.float32, shape=[None, self.data.train_graphs3d[0].shape[0], None, None, None])
        net3d = eq.equi_3_to_1('tri_equi0', self.data.train_graphs3d[0].shape[0], self.config.architecture3d[0],
                               self.graphs3)
        net3d = tf.nn.relu(net3d, name='rel3d')
        self.graphs2 = tf.placeholder(tf.float32, shape=[None, self.data.train_graphs2d[0].shape[0], None, None])
        net2d = eq.equi_2_to_2('2d_equ0', self.data.train_graphs2d[0].shape[0], self.config.architecture2d[0],
                               self.graphs2)
        net2d = tf.nn.relu(net2d, name='rel2d')
        for layer in range(1, len(self.config.architecture2d)):  # architecture is # of features of each layers
            net2d = eq.equi_2_to_2('2d_equ%d' % layer, self.config.architecture2d[layer - 1],
                                   self.config.architecture2d[layer], net2d)
            net2d = tf.nn.relu(net2d, name='2d_rel%d' % layer)
        net122 = eq.equi_1_to_2('122', self.data.train_graphs1d[0].shape[0], self.config.architecture1d[0], self.graphs1)
        net122 =  tf.nn.relu(net122, name='rel122')
        net2d = tf.concat([net2d, net122], axis=1)

        net2d = layers.diag_offdiag_maxpool(net2d)  # invariant max layer according to the invariant basis

        net221 = eq.equi_2_to_1('221', 4, self.config.architecture2d[0], self.graphs2[:,2:, :,:])
        net221 = tf.nn.relu(net221, name = 'relu221')

        net1d = eq.equi_1_to_1('1d_equ0', self.data.train_graphs1d[0].shape[0], self.config.architecture1d[0],
                               self.graphs1)
        net1d = tf.nn.relu(net1d, name='rel1')

        for layer in range(1, len(self.config.architecture2d)):  # architecture is # of features of each layers
            net1d = eq.equi_1_to_1('1d_equ%d' % layer, self.config.architecture1d[layer - 1],
                                   self.config.architecture2d[layer], net1d)
            net1d = tf.nn.relu(net1d, name='1d_rel%d' % layer)
        net1d = tf.concat([net1d, net221], axis=1)

        net1d = tf.reduce_sum(net1d, axis=2)
        net = tf.concat([net1d, net2d, net3d], axis=1)

        net = layers.fully_connected(net, self.config.fc[0], "full1")
        net = layers.fully_connected(net, self.config.fc[1], "full2")
        net = layers.fully_connected(net, self.config.fc[1], "full3")

        net = layers.fully_connected(net, 1, "fully4", activation_fn=None)      # regression
        # define loss function
        with tf.name_scope("loss"):
            distances = tf.losses.absolute_difference(labels=self.labels, predictions=net,
                                                      reduction=tf.losses.Reduction.NONE)

            self.loss = tf.reduce_sum(distances, axis=0)
            self.correct_predictions = tf.reduce_sum(distances, axis=0)
            self.pred = net
        # get learning rate with decay every 20 epochs
        learning_rate = self.get_learning_rate(self.global_step_tensor, self.data.train_size * self.config.decay_epoch)

        # choose optimizer
        if self.config.optimizer == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=self.config.momentum)
        elif self.config.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate)

        # define train step
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def get_learning_rate(self, global_step, decay_step):
        """
        helper method to fit learning rat
        :param global_step: current index into dataset, int
        :param decay_step: decay step, float
        :return: output: N x S x m x m tensor
        """
        learning_rate = tf.train.exponential_decay(
            self.config.learning_rate,  # Base learning rate.
            global_step * self.config.batch_size,
            decay_step,
            self.config.decay_rate,  # Decay rate.
            staircase=True)
        learning_rate = tf.maximum(learning_rate, 0.00001)
        return learning_rate


class QM9_invariant_basic_gnn3(BaseModel):
    def __init__(self, config, data):
        super(QM9_invariant_basic_gnn3, self).__init__(config)
        self.data = data
        self.build_model()
        self.init_saver()

    def build_model(self):
        # here you build the tensorflow graph of any model you want and define the loss.
        self.is_training = tf.placeholder(tf.bool)

        self.labels = tf.placeholder(tf.int32, shape=[None,1])
        ## for A^3
        if self.config.input_order == 3:
            self.graphs1 = tf.placeholder(tf.float32, shape=[None, self.data.train_graphs3d[0].shape[0], None, None, None])
            net3d = eq.equi_3_to_1('tri_equi0', self.data.train_graphs3d[0].shape[0], self.config.architecture3d[0], self.graphs1)
            net3d = tf.nn.relu(net3d, name='rel0')
            if self.config.network == 'gnn3':
                self.graphs2 = tf.placeholder(tf.float32, shape=[None, self.data.train_graphs[0].shape[0], None, None])
                net2d = eq.equi_2_to_2('2d_equ0', self.data.train_graphs[0].shape[0], self.config.architecture2d[0],
                                       self.graphs2)
                net2d = tf.nn.relu(net2d, name='rel0')
                for layer in range(1, len(self.config.architecture2d)):  # architecture is # of features of each layers
                    net2d = eq.equi_2_to_2('2d_equ%d' % layer, self.config.architecture2d[layer - 1],
                                           self.config.architecture2d[layer], net2d)
                    net2d = tf.nn.relu(net2d, name='2d_rel%d' % layer)
                net2d = layers.diag_offdiag_maxpool(net2d)  # invariant max layer according to the invariant basis
                net = tf.concat([net2d, net3d], axis=1)

            elif self.config.network == 'gnn4':
                self.graphs2 = tf.placeholder(tf.float32, shape=[None, self.data.train_graphs[0].shape[0], None])
                net1d = eq.equi_1_to_1('2d_equ0', self.data.train_graphs[0].shape[0], self.config.architecture1d[0],
                                       self.graphs2)
                net1d = tf.nn.relu(net1d, name='rel0')
                for layer in range(1, len(self.config.architecture2d)):  # architecture is # of features of each layers
                    net1d = eq.equi_1_to_1('2d_equ%d' % layer, self.config.architecture1d[layer - 1],
                                           self.config.architecture2d[layer], net1d)
                    net1d = tf.nn.relu(net1d, name='2d_rel%d' % layer)
                net1d = tf.reduce_sum(net1d,axis=2)
                net = tf.concat([net1d, net3d], axis=1)

        ## for A^4
        elif self.config.input_order == 4:
            self.graphs1 = tf.placeholder(tf.float32, shape=[None, 1, None, None, None, None])
            net4d = eq.equi_4_to_1('four_equi0', 1, self.config.architecture[0], self.graphs1)
            net4d = tf.nn.relu(net4d, name='rel0')
            self.graphs2 = tf.placeholder(tf.float32, shape=[None, self.data.train_graphs[0].shape[0], None, None])
            net2d = eq.equi_2_to_2('2d_equ0', self.data.train_graphs[0].shape[0], self.config.architecture2d[0],
                                   self.graphs2)
            net2d = tf.nn.relu(net2d, name='rel0')
            for layer in range(1, len(self.config.architecture2d)):  # architecture is # of features of each layers
                net2d = eq.equi_2_to_2('2d_equ%d' % layer, self.config.architecture2d[layer - 1],
                                       self.config.architecture2d[layer], net2d)
                net2d = tf.nn.relu(net2d, name='2d_rel%d' % layer)
            net2d = layers.diag_offdiag_maxpool(net2d)  # invariant max layer according to the invariant basis

            net = tf.concat([net2d, net4d], axis=1)


        net = layers.fully_connected(net, self.config.fc[0], "full1")
        net = layers.fully_connected(net, self.config.fc[1], "full2")

        net = layers.fully_connected(net, 1, "fully3", activation_fn=None)  # regression
        # define loss function
        with tf.name_scope("loss"):
            distances = tf.losses.absolute_difference(labels=self.labels, predictions=net,
                                                  reduction=tf.losses.Reduction.NONE)

        self.loss = tf.reduce_sum(distances, axis=0)
        self.correct_predictions = tf.reduce_sum(distances, axis=0)
        self.pred = net
        learning_rate = self.get_learning_rate(self.global_step_tensor, self.data.train_size * 20)

        # choose optimizer
        if self.config.optimizer == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=self.config.momentum)
        elif self.config.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate)

        # define train step
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def get_learning_rate(self, global_step, decay_step):
        """
        helper method to fit learning rat
        :param global_step: current index into dataset, int
        :param decay_step: decay step, float
        :return: output: N x S x m x m tensor
        """
        learning_rate = tf.train.exponential_decay(
            self.config.learning_rate,  # Base learning rate.
            global_step * self.config.batch_size,
            decay_step,
            self.config.decay_rate,  # Decay rate.
            staircase=True)
        learning_rate = tf.maximum(learning_rate, 0.00001)
        return learning_rate
