# 建構卷積層神經網路層 CNN

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    # with tf.name_scope('inputs'):
    #     with tf.name_scope('weights'):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # here to dropout
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:  # 如果激活函數是線性函數
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    tf.summary.histogram(layer_name + '/ouputs', outputs)
    return outputs


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):  # 通常bias用正值
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):  # x:輸入的數值、圖片值，W：權重
    # Must have strides[0] = strides[3] = 1 , strides = [1, stride, stride, 1]
    #  strides = [1, x_movement, y_movement, 1]

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2X2(x):
    # Must have strides[0] = strides[3] = 1
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784], name='x_input')  # 28X28，xs 是所有圖片例子
ys = tf.placeholder(tf.float32, [None, 10], name='y_input')
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# shape = [-1, 28, 28,1]，shape[0]表示先不管圖片維度，shape[3]表示圖片通道, 1:黑白，3:彩色

# print(x_image.shape) # [n_samples, 28,28,1]


## conv1 + max pooling layer ##
W_conv1 = weight_variable([5, 5, 1, 32])  # patch 5X5, in size 1 是圖片厚度, out size 32是輸出高度
b_conv1 = bias_variable([32])  # 對應輸出厚度 32
# conv2d(x_image, W_conv1) + b_conv1 與之前類似
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28x28x32
h_pool1 = max_pool_2X2(h_conv1)  # output size (28/2)x(28/2)x32

## conv2 + max pooling layer ##
W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5X5, in size 32 是圖片厚度, out size 64是輸出高度
b_conv2 = bias_variable([64])  # 對應輸出厚度 64
# conv2d(x_image, W_conv1) + b_conv1 與之前類似
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x64
h_pool2 = max_pool_2X2(h_conv2)  # output size 7x7x64

## fc1 layer ##
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7,7,64] ->> [n_samples,7*7*64 ]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# avoid overfitting
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# # avoid overfitting
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)




# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # loss
tf.summary.scalar('loss', cross_entropy)
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 巨大系統用AdamOptimizer比GradientDescent好
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)



sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)


# summary writer goes in here
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)


for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))