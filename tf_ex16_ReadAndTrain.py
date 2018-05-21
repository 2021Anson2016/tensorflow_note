import os
import random
import pandas as pd

import tensorflow as tf
import time
import numpy as np
from PIL import Image


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),

                                       })

    img = tf.decode_raw(features['img_raw'], tf.int64)
    # img = tf.reshape(img, [64, 64, 1])  # 一定要加，不然shape和ndim，會在read出錯
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label


cwd = os.getcwd()
CSV_DATA = pd.read_csv("class2.csv")  # ''内写入文件所在的详细目录
if CSV_DATA is None:
    print("csvData not exits")
else:
    print("exist")

label = '2'
class_path = cwd + "/" + label
num = 0
trainMatrix = []
# trainMatrix = np.array(shape=( 1, 7000, 4096, 1), )
for img_name in os.listdir(class_path):
    img_path = class_path + "/" + img_name
    img = np.asarray(Image.open(img_path))
    # img = img.resize((224, 224))
    num = num + 1
    print("num: ", num)
    print("label: ", label)
    print("img_name: ", img_name)
    img = img.resize((1, 4096))
    trainMatrix.append(img)




# def read_img(csvData):
#     num = 0
#
#     if csvData is None:
#         print("csvData not exits")
#     else:
#         print("exist")
#     for name in enumerate(csvData):
#         label = name[1]
#         class_path = cwd + "/" + name[1]
#         if int(name[1])<=csvData.size:
#             for img_name in os.listdir(class_path):
#                 img_path = class_path + "/" + img_name
#                 img = Image.open(img_path)
#                 # img = img.resize((224, 224))
#                 num = num + 1
#                 print("num: ", num)
#                 print("label: ", label)
#                 print("img_name: ", img_name)
#     return img, label


# image, label = read_img(CSV_DATA)


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


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


# img, ll = read_and_decode("train_class0.tfrecords")
# for _ in range()
# label = tf.zeros([7000, 1], dtype=np.int32)
# label = np.asarray([tf.zeros([7000,0])+1])
label = []

# for i in range(7000):
#     label.append([0])

label = np.asarray(label)

# filename_queue = tf.train.string_input_producer(["train_class0.tfrecords"],) #读入流中
# reader = tf.TFRecordReader()
# _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
# features = tf.parse_single_example(serialized_example,
#                                        features={
#                                            'label': tf.FixedLenFeature([], tf.int64),
#                                            'img_raw' : tf.FixedLenFeature([], tf.string),
#                                        })  #取出包含image和label的feature对象





batch_size = 128
def next_batch(imgs, labels, size):
    id_samp = np.ndarray(shape=(size), dtype=np.int32)
    img_samp = np.ndarray(shape=(size, imgs.shape[1], imgs.shape[2]))
    label_samp = np.ndarray(shape=(size, labels.shape[1]))
    for i in range(size):
        r = random.randint(0,imgs.shape[0]-1)
        img_samp[i] = imgs[r]
        label_samp[i] = labels[r]
        id_samp[i] = r
    return [img_samp, label_samp]

channel_size = 1

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [batch_size, 4096, channel_size], name='x_input')  # 64X64，xs 是所有圖片例子
ys = tf.placeholder(tf.float32, [batch_size, 1], name='y_input')  # 1:label size
# keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [batch_size, 64, 64, 1])

## conv1 + max pooling layer ##
W_conv1 = weight_variable([5, 5, 1, 32])  # patch 5X5, in size 1 是圖片厚度, out size 32是輸出高度
b_conv1 = bias_variable([32])  # 對應輸出厚度 32
# conv2d(x_image, W_conv1) + b_conv1 與之前類似
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 64x64x32
h_pool1 = max_pool_2X2(h_conv1)  # output size 32x32x32

## conv2 + max pooling layer ##
W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5X5, in size 32 是圖片厚度, out size 64是輸出高度
b_conv2 = bias_variable([64])  # 對應輸出厚度 64
# conv2d(x_image, W_conv1) + b_conv1 與之前類似
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 32x32x64
h_pool2 = max_pool_2X2(h_conv2)  # output size 7x7x64

## fc1 layer ##
W_fc1 = weight_variable([16 * 16 * 64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7,7,64] ->> [n_samples,7*7*64 ]
h_pool2_flat = tf.reshape(h_pool2, [batch_size, 16 * 16 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# avoid overfitting
h_fc1_drop = tf.nn.dropout(h_fc1, 1)

## fc2 layer ##
W_fc2 = weight_variable([1024, 3])
b_fc2 = bias_variable([3])

prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# # avoid overfitting
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)




# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # loss
tf.summary.scalar('loss', cross_entropy)

# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 巨大系統用AdamOptimizer比GradientDescent好
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 初始化所有的op
init = tf.global_variables_initializer()
saver = tf.train.Saver()

sess = tf.Session()
sess.run(init)
coord = tf.train.Coordinator()

# summary writer goes in here
train_writer = tf.summary.FileWriter("logs_anson/train", sess.graph)
test_writer = tf.summary.FileWriter("logs_anson/test", sess.graph)

for i in range(1000):
    # print(i)
    # 启动队列
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # for i in range(3):
    #     val, l= sess.run([img_batch, label_batch])
    #     #l = to_categorical(l, 12)
    #     print(val.shape, l)
    read_tfrecord_start_time = time.time()
    # try:
    #     while not coord.should_stop():
    #         imgs = sess.run([img_batch])
    #         for img in imgs:
    #             print(img.shape)
    # except Exception as e:
    #     coord.request_stop(e)
    # finally:
    #     coord.request_stop()
    # coord.join(threads)
    # read_tfrecord_duration = time.time() - read_tfrecord_start_time
    # print("Read TFrecord Duration:   %.3f" % read_tfrecord_duration)


    img_batch, label_batch =  next_batch(trainMatrix,label,batch_size)
    sess.run(train_step, feed_dict={xs: img_batch, ys: label_batch})
    # print( sess.run(train_step, feed_dict={xs: img_batch, ys: label_batch}))
    if i % 50 == 0:
        print(sess.run(cross_entropy, feed_dict={xs: img_batch, ys: label_batch}))

coord.request_stop()
coord.join(threads)

saver.restore(sess, "my-test-model/anson_net.ckpt")
saver.save(sess, 'my-test-model')
