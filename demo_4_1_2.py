#encoding=utf-8
#COST=1 PROFIT=9

import tensorflow as tf
import numpy as np

BATCH_SIZE=8
SEED=23455
learning_rate=0.001
epoch=20001
cost=1
profit=9

rdm=np.random.RandomState(SEED)
b=rdm.rand()/10.0-0.05
X=rdm.rand(32,2)
Y=[[x1+x2+b] for (x1,x2) in X]

x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1))
w1=tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y=tf.matmul(x,w1)

loss=tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*cost,(y_-y)*profit))
# loss=tf.reduce_mean(tf.square(y_-y))
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(epoch):
        start=(i*BATCH_SIZE)%32
        end=start+BATCH_SIZE
        train_x = X[start:end]
        train_y = Y[start:end]
        sess.run(train_step,feed_dict={x:train_x,y_:train_y})

        if i%500==0:
            total_loss=sess.run(loss,feed_dict={x:X,y_:Y})
            print("epoch: ",i ," total loss: ",total_loss)
    print("Final w1：",sess.run(w1))