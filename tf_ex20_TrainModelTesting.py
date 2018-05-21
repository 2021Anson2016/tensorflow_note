import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import math
import random
import cv2


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def LoadDataFileFolder_gray(path, total):
    files = [f for f in os.listdir(path)]
    img_list = []
    label_list = []
    count = 0

    for filename in files:
        if count > total and total > 0:
            break
        f = filename.split(".")

        if len(f) == 2 and f[1].strip() == "jpg":
            if filename[0] == 'a':
                label_list.append([1, 0, 0])
            elif filename[0] == 'b':
                label_list.append([0, 1, 0])
            else:
                label_list.append([0, 0, 1])

            img = np.asarray(Image.open(path + "/" + filename))
            gray = rgb2gray(img).tolist()
            img_list.append(img)
            count += 1

    img_list = np.asarray(img_list)
    label_list = np.asarray(label_list)
    return img_list, label_list

def LoadDataFileFolder_RGB(path, total):
    files = [f for f in os.listdir(path)]
    img_list = []
    label_list = []
    count = 0

    for filename in files:
        if count > total and total > 0:
            break
        f = filename.split(".")

        if len(f) == 2 and f[1].strip() == "jpg":
            if filename[0] == 'a':
                label_list.append([1, 0, 0])
            elif filename[0] == 'b':
                label_list.append([0, 1, 0])
            else:
                label_list.append([0, 0, 1])

            img = np.asarray(Image.open(path + "/" + filename))
            gray = rgb2gray(img).tolist()
            img_list.append(gray)
            count += 1

    img_list = np.asarray(img_list)
    label_list = np.asarray(label_list)
    return img_list, label_list


path_train = "3class_training data"
path_test = "testing"
imgs_train, labels_train = LoadDataFileFolder_RGB(path_train, -1)
imgs_test, labels_test = LoadDataFileFolder_RGB(path_test, -1)
WINDOW_NAME = ("WEBCAM")
webcam = cv2.VideoCapture(0)

width = 64
height = 64
batch_size = 128


def next_batch(imgs, labels, size):
    id_samp = np.ndarray(shape=(size), dtype=np.int32)
    img_samp = np.ndarray(shape=(size, imgs.shape[0], imgs.shape[1]))
    label_samp = np.ndarray(shape=(size, labels.shape[1]))
    for i in range(size):
        r = random.randint(0, imgs.shape[0] - 1)
        img_samp[i] = imgs[r]
        label_samp[i] = labels[r]
        id_samp[i] = r
    return [img_samp, label_samp]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):  # 通常bias用正值
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, strides):  # x:輸入的數值、圖片值，W：權重
    # Must have strides[0] = strides[3] = 1 , strides = [1, stride, stride, 1]
    #  strides = [1, x_movement, y_movement, 1]
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')


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


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [batch_size, 64, 64], name='x_input')  # 64X64，xs 是所有圖片例子
xs2 = tf.placeholder(tf.float32, [64, 64], name='x_input2')  # 64X64，xs 是所有圖片例子

ys = tf.placeholder(tf.float32, [batch_size, 3], name='y_input')  # 3:label size
ys2 = tf.placeholder(tf.float32, [1, 3], name='y_input2')  # 3:label size

keep_prob = tf.placeholder(tf.float32)
xs_re = tf.reshape(xs, [batch_size, 64, 64, 1])
xs2_re = tf.reshape(xs2, [1, 64, 64, 1])

## conv1 + max pooling layer ##
W_conv1 = weight_variable([5, 5, 1, 32])  # patch 5X5, in size 1 是圖片厚度, out size 32是輸出高度
b_conv1 = bias_variable([32])  # 對應輸出厚度 32
# conv2d(x_image, W_conv1) + b_conv1 與之前類似
h_conv1 = tf.nn.relu(conv2d(xs_re, W_conv1, [1, 1, 1, 1]) + b_conv1)  # output size batch_sizex64x64x32：
h_conv1_test = tf.nn.relu(conv2d(xs2_re, W_conv1, [1, 1, 1, 1]) + b_conv1)  # output size 1x64x64x32：

h_pool1 = max_pool_2X2(h_conv1)  # output size 32x32x32
h_pool1_test = max_pool_2X2(h_conv1_test)  # output size 32x32x32

## conv2 + max pooling layer ##
W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5X5, in size 32 是圖片厚度, out size 64是輸出高度
b_conv2 = bias_variable([64])  # 對應輸出厚度 64

# conv2d(x_image, W_conv1) + b_conv1 與之前類似
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, [1, 1, 1, 1]) + b_conv2)  # output size 32x32x64
h_conv2_test = tf.nn.relu(conv2d(h_pool1_test, W_conv2, [1, 1, 1, 1]) + b_conv2)  # output size 32x32x64

h_pool2 = max_pool_2X2(h_conv2)  # output size 16x16x64
h_pool2_test = max_pool_2X2(h_conv2_test)  # output size 16x16x64

## fc1 layer ##
W_fc1 = weight_variable([16 * 16 * 64, 1024])
b_fc1 = bias_variable([1024])

# [n_samples, 7,7,64] ->> [n_samples,7*7*64 ]
h_pool2_flat = tf.reshape(h_pool2, [batch_size, 16 * 16 * 64])
h_pool2_flat_test = tf.reshape(h_pool2_test, [1, 16 * 16 * 64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_test = tf.nn.relu(tf.matmul(h_pool2_flat_test, W_fc1) + b_fc1)

# # avoid overfitting
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# avoid overfitting
# h_fc1_drop = tf.nn.dropout(h_fc1, 1)

## fc2 layer ##
W_fc2 = weight_variable([1024, 3])
b_fc2 = bias_variable([3])

## ----------------------------------------------------------------------------------------------
# prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# prediction = tf.matmul(h_fc1, W_fc2) + b_fc2
prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

prediction_test = tf.matmul(h_fc1_test, W_fc2) + b_fc2

y_prob = tf.nn.sigmoid(prediction)
y_prob_test = tf.nn.sigmoid(prediction_test)




# the error between prediction and real data
# cross_entropy = tf.reduce_mean(tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))  # loss
# cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, targets=ys), 1))
# 參數要給labels ，不是targets

cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=ys), 1))
cross_entropy2 = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction_test, labels=ys2), 1))

tf.summary.scalar('loss', cross_entropy)

# train_op  = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 巨大系統用AdamOptimizer比GradientDescent好
train_op  = tf.train.AdamOptimizer().minimize(cross_entropy)

# 初始化所有的op

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# summary writer goes in here
train_writer = tf.summary.FileWriter("save/train_graph/", sess.graph)
test_writer = tf.summary.FileWriter("save/test_graph/", sess.graph)

if not os.path.exists('save/'):
    os.makedirs('save/')

isTrain = True
isTest = True


AccNum = 0
labels_test = [[0, 0, 0]]


def maxPos(a):
    m=max(a)
    for i, j in enumerate(a):
        if j == m:
            return i


RE_NAME = "reGrayImg"

print("訓練中！")
for step in range(3000):
    if isTrain:  # training
        img_batch, label_batch = next_batch(imgs_train, labels_train, batch_size)
        # y_train, loss_ = sess.run([train_op , cross_entropy], feed_dict={xs: img_batch, ys: label_batch})
        y_train, loss_ = sess.run([train_op, cross_entropy], feed_dict={xs: img_batch, ys: label_batch, keep_prob: 0.6})

        if step % 50 == 0 and step != 0:
            # img_batch, label_batch = next_batch(imgs_train, labels_train, batch_size)
            y__, loss_ = sess.run([y_prob, cross_entropy], feed_dict={xs: img_batch, ys: label_batch, keep_prob: 0.6})
            print("Training step: " + str(step ) + " " + str(loss_))
            print(y__[0])
            print(label_batch[0])
            AccNum = 0
            for num in range(0, 128, 1):
                if (y__[num] == label_batch[num]).all() or maxPos(y__[num])==maxPos(label_batch[num]):
                    # print("y__[num] ", y__[num])
                    # print("label_batch[num] ", label_batch[num])

                    AccNum = AccNum + 1
            print(AccNum)
            print("準確%", (AccNum / 128) * 100)

saveName = "model.ckpt"
saver = tf.train.Saver()
save_path = saver.save(sess, "save/" + saveName)
print("訓練完成！")

print("保存模型成功！")
print("Model saved in file: %s" % save_path)

count = 0

if isTest:
    while 1:
        ret, Shooting_images = webcam.read()
        cv2.imshow(WINDOW_NAME, Shooting_images)
        img_re = cv2.resize(Shooting_images, (width, height))
        img_reGray = rgb2gray(img_re)/255
        cv2.imshow(RE_NAME, img_reGray)

        y__, loss_ = sess.run([y_prob_test, cross_entropy2], feed_dict={xs2: img_reGray, ys2: labels_test, keep_prob: 1})
        print(y__[0])
        if (y__[0] == [1, 0, 0]).all() or maxPos(y__[0]) == 0:
            print("布")
        elif (y__[0] == [0, 1, 0]).all() or maxPos(y__[0]) == 1:
            print("石頭")
        else:
            print("食指")


        count += 1
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        # press ESC leave
    cv2.destroyAllWindows()


