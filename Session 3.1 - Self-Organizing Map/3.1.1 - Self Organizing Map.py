import tensorflow as tf
import numpy as np

class SOM:
    def __init__(self, width, height, in_dim):
        self.width = width
        self.height = height
        self.in_dim = in_dim

        node = [[x, y] 
            for y in range(height) 
            for x in range(width)]
        self.node = tf.to_float(node)

        self.weight = tf.Variable(tf.random_normal([width * height, in_dim]))

        self.x = tf.placeholder(tf.float32, [in_dim])

        # bmu -> best matching unit
        winning_node = self.get_bmu(self.x)

        self.update = self.update_neighbor(winning_node, self.x)
    
    def get_bmu(self, x):
        expanded_x = tf.expand_dims(x, 0)
        square_diff = tf.square(tf.subtract(expanded_x, self.weight))
        dists = tf.reduce_sum(square_diff, 1)
        winner_index = tf.argmin(dists, 0)
        winner_loc = tf.stack(
            [tf.mod(winner_index, self.width), 
            tf.div(winner_index, self.width)])
        return winner_loc

    def update_neighbor(self, winning_node, x):
        winning_node = tf.to_float(winning_node)
        lr = .5
        sigma = tf.to_float(tf.maximum(self.width, self.height)) / 2.
        expanded_bmu = tf.expand_dims(winning_node, 0)
        sqr_diff_from_winner = tf.square(tf.subtract(expanded_bmu, self.node))
        sqr_dist_from_winner = tf.reduce_sum(sqr_diff_from_winner, 1)
        neighbor_strength = tf.exp(-tf.div(sqr_dist_from_winner, 2. * tf.square(sigma)))
        rate = tf.multiply(lr, neighbor_strength)
        number_of_nodes = self.width * self.height
        rf = tf.stack([tf.tile(tf.slice(rate, [i], [1]), [self.in_dim]) for i in range(number_of_nodes)])
        x_w_diff = tf.subtract(tf.stack([x for i in range(number_of_nodes)]), self.weight)
        weight_diff = tf.multiply(rf, x_w_diff)
        update_node = tf.add(self.weight, weight_diff)
        return tf.assign(self.weight, update_node)

    def train(self, dataset, number_of_epoch):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(number_of_epoch):
                for data in dataset:
                    feed = {self.x: data}
                    sess.run([self.update], feed)
            self.weight_val = list(sess.run(self.weight))
            self.node_val = list(sess.run(self.node))
            cluster = [[] for i in range(self.width)]
            for i, location in enumerate(self.node_val):
                cluster[int(location[0])].append(self.weight_val[i])
            self.cluster = cluster

from matplotlib import pyplot as plt
colors = np.array(
    [
        [0., 0., 0.],
        [1., 1., 1.],
        [.3, .3, .3],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 1., 0.],
        [0., 1., 1.],
        [1., 0., 1.],
        [.3, .4, .5],
        [.2, .2, .2],
    ]
)

som = SOM(4, 4, 3)
som.train(colors, 100)

plt.imshow(som.cluster)
plt.show()