#  變數練習
import tensorflow as tf

state = tf.Variable(0, name='counter')  # tensorflow 語法是必須定義為變量才能事變量
# print(state.name)
one = tf.constant(1)

new_value = tf.add(state , one)
update = tf.assign(state, new_value)
init = tf.initialize_all_variables()  # must have if define variable " 變數必須要被初始化 "

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))