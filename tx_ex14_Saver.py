# 保存和存取

import tensorflow as tf
import numpy as np
## Save to file
# remember to define the same dtype and shape when restore
# W = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name='weights')
# b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biaes')
#
# init = tf.initialize_all_variables()
# saver = tf.train.Saver()
#
# with tf.Session()as sess:
#     sess.run(init)
#     saver_path = saver.save(sess, "my_net/save_net.ckpt")
#     print("Save to path:", saver_path)


## restore variables
# refine the same shape and same type for your variable
# W = tf.Variable(tf.zeros())
W = tf.Variable(np.arange(6).reshape((2,3)), dtype=tf.float32, name='weights')
b = tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32, name='biaes')

# not need init step
saver = tf.train.Saver() # 做提取文件或儲存文件
with tf.Session()as sess:
    saver.restore(sess, "my_net/save_net.ckpt")
    print("weights:", sess.run(W))
    print("biaes:", sess.run(b))

# 目前tensorflow 只能存變數，還不能儲存全部的神經網路框架，
# 所以必須要重新定義框架，再把變數載入