import tensorflow as tf
import numpy as np
#import tensorflow.keras.datasets.mnist

def convolution(inp, nodes_in, nodes_out):
    w = tf.Variable(tf.random_normal([3, 3, nodes_in, nodes_out])) # 4 dimensi
    b = tf.Variable(tf.random_normal([nodes_out]))

    conv = tf.nn.conv2d(inp, w, strides=[1, 1, 1, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    mp = tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    return mp

def fully_connected(inp, nodes_in, nodes_out):
    w = tf.Variable(tf.random_normal([nodes_in, nodes_out]))
    b = tf.Variable(tf.random_normal([nodes_out]))
    linear_comb = tf.matmul(inp, w) + b
    return linear_comb

x = tf.placeholder(tf.float32, [None, 28 * 28])
y = tf.placeholder(tf.float32, [None, 10])

def build_model(x):
    x_as_img = tf.reshape(x, [-1, 28, 28, 1])
    conv1 = convolution(x_as_img, 1, 16)
    conv2 = convolution(conv1, 16, 8)
    flatten = tf.reshape(conv2, [-1, 7 * 7 * 8])
    fc = fully_connected(flatten, 7 * 7 * 8, 10)
    return fc

lr = .1
number_of_iter = 1000
report_between = int(.1 * number_of_iter)

def optimize(model, dataset):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100

    mnist_ds = tf.data.Dataset.from_tensor_slices(dataset)
    mnist_ds.shuffle(1000)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(number_of_iter):
            batched = mnist_ds.batch(100)
            iterator = batched.make_one_shot_iterator()
            x_feed, y_feed = iterator.get_next()
            x_feed = tf.to_float(tf.reshape(x_feed, [-1, 784]))
            x_feed_val = sess.run(x_feed)
            y_feed_val = sess.run(y_feed) # 0 - 9
            y_feed_one_hot = []
            for val in y_feed_val:
                one_hot = np.zeros([10], 'int')
                one_hot[val] = 1
                y_feed_one_hot.append(one_hot)
            feed = {x: x_feed_val, y: y_feed_one_hot}
            _, loss_val, acc_val = sess.run([optimizer, loss, accuracy], feed)

            if i % report_between == 0:
                print("[Epoch: {:5} Loss {:1.8f} Accuracy {:3.2f}%".format(i, loss_val, acc_val))

mnist = tf.keras.datasets.mnist.load_data()
dataset = mnist[0] # return (image, label)
model = build_model(x)
optimize(model, dataset)
