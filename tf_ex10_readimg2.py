import tensorflow as tf
import os
import time
from PIL import Image
import imageflow


root = "./training2/"

def get_filenames(path):
    filenames = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if ".jpg" in f:
                filenames.append(os.path.join(root, f))
    return filenames

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def read_img(filenames, num_epochs, shuffle=True):
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TFRecordReader()
    key, value = reader.read(filename_queue)
    img = tf.image.decode_jpeg(value, channels=1)

    for name in filenames:
        img = Image.open(name)
        # print(img.mode)  #
        if img.mode == "RGB":
            img_raw = img.tobytes()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[name])),
                        "img_raw": tf.train.Feature( bytes_list=tf.train.BytesList(value=[img_raw]))} ))
        else:
            # print("gray image")
            img_raw = img.tobytes()   #将图片转化为原生bytes
            print(name)
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(name)])),
                    "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])) }))
    return img

FileName = get_filenames(root)
print(FileName)
image = read_img(FileName, 1, True )
print(image.index())

# convert_images(image, labels, FileName)

# images and labels array as input
def convert_to_tfrecord():
    writer = tf.python_io.TFRecordWriter("training.tfrecords")
    filenames = get_filenames(root)
    with tf.device('/gpu:2'):
        for name in filenames:
            img = Image.open(name)
            if img.mode == "RGB":
                img = img.resize((64, 64), Image.NEAREST)
                img_raw = img.tobytes()

                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        # "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[name])),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    }
                   ))
                writer.write(example.SerializeToString())
            else:
                # print("gray image")
                # img_raw = img.tobytes()   #将图片转化为原生bytes

                # 将图像矩阵转化成一个字符串
                # image_raw = img[name].tostring()
                img_raw = img.tobytes()
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        "label": tf.train.Feature(int64_list=tf.train.int64_list(value=[name])),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    }
                ))
                writer.write(example.SerializeToString())  #序列化为字符串

    writer.close()


# init_op = tf.global_variables_initializer()
# with tf.Session() as sess:
#   sess.run(init_op)

# Start populating the filename queue.
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess, coord=coord)


# create_tfrecord_start_time = time.time()
# convert_to_tfrecord()






# for serialized_example in tf.python_io.tf_record_iterator("./training.tfrecords"):
#     example = tf.train.Example()
#     # if example is None:
#     #     print("Empty")
#     # else:
#     #     print("ok")
#     #
#     # if serialized_example == [] :
#     #     print("Empty")
#     # else:
#     #     print("ok")
#
#     example.ParseFromString(serialized_example)
#
#     image = example.features.feature['image'].int64_list.value
#     # label = example.features.feature['label'].int64_list.value
#     # 可以做一些预处理之类的
#     print ("image", image)
#     # print("label", label)
