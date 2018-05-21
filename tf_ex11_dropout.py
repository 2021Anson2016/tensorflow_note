# 建構神經網路層
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# import tensorflow.examples.tutorials.mnist.input_data
from sklearn.datasets import load_digits
import sklearn.preprocessing.label
# load data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label
from sklearn.preprocessing import LabelBinarizer

digits = load_digits()
X = digits.data
y = digits.target
y = label.LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


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
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.arg_max(y_pre, 1), tf.arg_max(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


# # define placeholder for inputs to network
# with tf.name_scope('inputs'):
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64], name='x_input')  # 28X28
# print(xs)
ys = tf.placeholder(tf.float32, [None, 10], name='y_input')

# add output layer
l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)

prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # loss
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

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
    # sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})  # no dropout

    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob:0.6})  # drop 40%, keep_prob:0.6 == 保持60% 不被drop
    if i % 50 == 0:
        # record loss
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.6})
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 0.6})
        # train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train })
        # test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)
