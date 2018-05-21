# 建構神經網路層
import tensorflow as tf
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.examples.tutorials.mnist.input_data

# number 1 to 10 data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def add_layer(inputs, in_size, out_size, activation_fuction=None):
    with tf.name_scope('inputs'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_fuction is None:  # 如果激活函數是線性函數
        outputs = Wx_plus_b
    else:
        outputs = activation_fuction(Wx_plus_b)
    return outputs


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
    return result


# # define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 784], name='x_input')  # 28X28
    # print(xs)
    ys = tf.placeholder(tf.float32, [None, 10], name='y_input')

# add output layer
prediction = add_layer(xs, 784, 10, activation_fuction=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # loss

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.summary.scalar('loss', cross_entropy)



# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()

sess = tf.Session()
merged = tf.summary.merge_all()
# summary writer goes in here
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)

# important step
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)  # 從data base取100做訓練
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))


        # # writer = tf.train.SummaryWriter("View/", sess.graph)
        # # tf.train.SummaryWriter soon be deprecated, use following
        # if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
        #     writer = tf.train.SummaryWriter('logs/', sess.graph)
        # else: # tensorflow version >= 0.12
        #     writer = tf.summary.FileWriter("logs/", sess.graph)
