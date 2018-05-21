import tensorflow as tf
import os

import time
from PIL import Image

root = "./training2/"


def get_filenames(path):
    filenames = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if ".jpg" in f:
                filenames.append(os.path.join(root, f))
    return filenames


def read_img(filenames, num_epochs, shuffle=True):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    img = tf.image.decode_jpeg(value, channels=1)
    #    img = tf.image.resize_images(img, size=(256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    for name in filenames:
        img = Image.open(name)
        print(img.mode)
        if img.mode == "RGB":
            img_raw = img.tobytes()
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    "img_raw": tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[img_raw]))}))
        else:
            print("gray image")
    return img


def convert_to_tfrecord():
    writer = tf.python_io.TFRecordWriter("./training.tfrecords")
    filenames = get_filenames(root)
    for name in filenames:
        img = Image.open(name)
        if img.mode == "RGB":
            img = img.resize((256, 256), Image.NEAREST)
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
    else:
        print("gray image")
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())

    writer.close()


img = read_img(get_filenames(root), 1, True)
create_tfrecord_start_time = time.time()
convert_to_tfrecord()

# with tf.Session() as sess:
#     # img = read_img(get_filenames(root), 1, True)
#     # print(img)
#     """读取图片文件"""
#     file1 = tf.read_file(root+'0.jpg')
#     """解码图片，png格式用tf.image.decode_png，
#     channels=3表示RGB，1表示灰度"""
#     image = tf.image.decode_jpeg(file1, channels=1)
#
#     # """调整图片大小，size=[new_height, new_width]"""
#     # image = tf.image.resize_images(image, size=[32, 32])
#     """转化图片转化为float32类型，并缩放到[0,1]之间，
#     也可使用 tf.cast(image, tf.float32)/255(一般图片类型最大值为255)"""
#     image = tf.image.convert_image_dtype(image, tf.float32)
#     sess.run(image)
#     print(image)
