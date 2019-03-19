#coding:utf-8
import tensorflow as tf
import numpy as np

BATCH_SIZE=8
SEED=23455
learning_rate=0.001

rng=np.random.RandomState(SEED)
#生成32组数据,每组两个特征

X=rng.rand(32,2)
Y=[[int(x0+x1<1)] for (x0,x1) in X]

print("X:\n",X)
print("Y:\n",Y)

x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1)) #预测值

w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

#反向传播
loss=tf.reduce_mean(tf.square(y-y_))
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#train_step=tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss)
#train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
	init_op=tf.global_variables_initializer()
	sess.run(init_op)
	print("W1:\n",sess.run(w1))
	print("W2:\n",sess.run(w2))

	epoch=10001
	for i in range(epoch):
		start=(i*BATCH_SIZE)%32
		end= start+BATCH_SIZE
		train_x=X[start:end]
		train_y=Y[start:end]

		sess.run(train_step,feed_dict={x:train_x,y_:train_y})
		if i% 500==0:
			total_loss=sess.run(loss,feed_dict={x:X,y_:Y})
			print("epoch:",i," loss:",total_loss)
	print("\n")
	print("w1:\n",sess.run(w1))
	print("w2:\n",sess.run(w2))