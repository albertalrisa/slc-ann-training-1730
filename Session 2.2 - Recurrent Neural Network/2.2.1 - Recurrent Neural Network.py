import tensorflow as tf
import numpy as np

class RNNModel:
    def __init__(self, batch_size, unroll_count, context_count, training=True):
        if not training:
            batch_size = 1
            unroll_count = 1
        
        self.cell = tf.nn.rnn_cell.BasicRNNCell(context_count)

        self.input = tf.placeholder(tf.float32, [batch_size, unroll_count, 1])
        self.target = tf.placeholder(tf.float32, [batch_size, unroll_count, 1])

        self.initial_state = self.cell.zero_state(batch_size, tf.float32)

        with tf.variable_scope('Wx_b'):
            w = tf.get_variable('W', [context_count, 1])
            b = tf.get_variable('b', [1])

        rnn_out, last_state = tf.nn.dynamic_rnn(self.cell, self.input, initial_state=self.initial_state)
        out = tf.reshape(tf.concat(rnn_out, 1), [-1, context_count])
        self.Wx_b = tf.matmul(out, w) + b
        self.act = tf.nn.sigmoid(self.Wx_b)

        flat_target = tf.reshape(tf.concat(self.target, 1), [-1])
        
        self.loss = tf.reduce_mean(.5 * (flat_target - self.act) ** 2)
        self.final_state = last_state

        self.lr = tf.Variable(0.0, trainable=False)

        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

dataset = [.4, .5, .6, .7, .8, .9, .8, .7, .6, .5, .4, .3, .2, .1, .2, .3, .4, .5, .6, .7]
batch_size = 4
unroll_count = 2
context_count = 3
number_of_epoch = 100
report_between = int(.1 * number_of_epoch)

save_dir = './rnn-model/'
filename = 'rnn.ckpt'

def optimize(model, batch_size, unroll_count, dataset):
    number_of_batch = len(dataset) // (batch_size * unroll_count)
    trim_data = number_of_batch * batch_size * unroll_count
    dataset = dataset[:trim_data]

    x_data = np.array(dataset)
    y_data = np.copy(x_data)
    y_data[:-1] = x_data[1:]
    y_data[-1] = x_data[0]

    x_batches = x_data.reshape([batch_size, unroll_count, -1])
    x_batches = np.split(x_batches, number_of_batch, 2)

    y_batches = y_data.reshape([batch_size, unroll_count, -1])
    y_batches = np.split(y_batches, number_of_batch, 2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.assign(model.lr, .1))

        saver = tf.train.Saver(tf.global_variables())

        for e in range(number_of_epoch):
            state = sess.run(model.initial_state)
            for b in range(number_of_batch):
                feed_x = x_batches[b]
                feed_y = y_batches[b]

                feed = {model.input: feed_x, \
                    model.target: feed_y, \
                    model.initial_state: state}
                
                loss_val, state, _ = \
                    sess.run([model.loss, model.final_state, model.optimizer], feed)

                batch_number = e * number_of_batch + b
                if batch_number % 50 == 0:
                    print('[Batch {:5}] Loss: {:1.8f}'.format(batch_number, loss_val))

                if batch_number % 10 == 0:
                    saver.save(sess, save_dir + filename, batch_number)

# model = RNNModel(batch_size, unroll_count, context_count)
# optimize(model, batch_size, unroll_count, dataset)

def sample(model):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

            while True:
                state = sess.run(model.initial_state)
                user_input = input('Input data: ').split(' ')
                user_input = [float(x) for x in user_input]
                for u_input in user_input:
                    x = np.array(u_input).reshape(1, 1, 1)
                    feed = {model.input: x, model.initial_state: state}
                    pred, state = sess.run([model.act, model.final_state], feed)
                print(pred)

model = RNNModel(batch_size, unroll_count, context_count, False)
sample(model)
