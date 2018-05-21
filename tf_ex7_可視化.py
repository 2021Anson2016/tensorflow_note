# 建構神經網路層
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_fuction=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_fuction is None:  # 如果激活函數是線性函數
        outputs = Wx_plus_b
    else:
        outputs = activation_fuction(Wx_plus_b)
    return outputs

# Make up some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise
## look at the value
# print(np.newaxis)
# # print(np.linspace(-1,1,300))
# print(x_data.shape)  #(50, 1)
# # print(x_data)


# 輸給train
xs = tf.placeholder(tf.float32, [None, 1])
# print(xs)
ys = tf.placeholder(tf.float32, [None, 1])
# add hidden layer l1
l1 = add_layer(xs, 1, 10, activation_fuction=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_fuction=None)

# tf.reduce_sum 求出loss 之和；  tf.reduce_mean：求出平均值
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

# GradientDescentOptimizer 裡面參數要給learning rate(<1) 0.1
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()

sess = tf.Session()
# 上面都沒任何運算，直到執行這行sess.run(init)
sess.run(init)


fig = plt.figure()
ax =fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()  # 連續的show 圖
plt.show()

for i in range(5000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

    if i % 50 == 0:
        # to visualize the result and improvement
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass

        prediction_value = sess.run(prediction, feed_dict={xs:x_data})
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5) # lw = line width
        plt.pause(0.5)
