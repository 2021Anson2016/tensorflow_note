# session 是執行命令
import tensorflow as tf
import numpy as np

#create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3

## create tensorflow struture start ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # 隨機產生 [-1.00]
biases = tf.Variable(tf.zeros([1])) # 初始值=0

y = Weights*x_data+biases

loss = tf.reduce_mean(tf.square(y-y_data))  #預測y與實際y_data 差值\
optimizer = tf.train.GradientDescentOptimizer(0.5)  #  梯地下降法 ，0.5 為學習效率

train = optimizer.minimize(loss)
init = tf.initialize_all_variables()
## create tensorflow struture start ###

sess = tf.Session() # 指標到要處理的地方
sess.run(init)  # 執行： 非常重要

for step in range(201):
    sess.run(train)
    if step%20==0:  # 每20步去顯示
        print(step, sess.run(Weights), sess.run(biases))
