import tensorflow as tf
import  numpy as np
from demo_4_5 import forward,generateds
from matplotlib import  pyplot as plt
REGULARIZER=0.01
LEARNING_RATE=0.001
EPOCH=20001
BATCH_SIZE=30
LEARNING_RATE_DECAY=0.999



def backward():
    x=tf.placeholder(tf.float32,shape=(None,2))
    y_=tf.placeholder(tf.float32,shape=(None,1))

    X, Y_, Y_C = generateds.generateds()
    y = forward.forward(X, REGULARIZER)
    global_step=tf.Variable(0,trainable=False)
    learning_rate=tf.train.exponential_decay(
        LEARNING_RATE,global_step,300/BATCH_SIZE,LEARNING_RATE_DECAY,
    staircase=True)

    loss_mse=tf.reduce_mean(tf.square(y-y_))
    loss_total = loss_mse + tf.add_n(tf.get_collection("losses"))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)

    with tf.Session() as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(EPOCH):
            start =(i*BATCH_SIZE)%300
            end=start+BATCH_SIZE
            train_x=X[start:end]
            train_y=Y_[start:end]
            sess.run(train_step,feed_dict={x:train_x,y_:train_y})

            if i%2000==0:
                loss_v=sess.run(loss,feed_dict={x:X,y_:Y_})
                print("i:{}  loss: {}".format(i,loss_v))
        xx,yy=np.mgrid[-3:3:.01,-3:3:.01]
        grid=np.c_[xx.ravel(),yy.ravel()]
        probs=sess.run(y,feed_dict={x,grid})
        probs=probs.reshape(xx.shape)
    plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_C))
    plt.scatter(xx,yy,probs,levels=[.5])
    plt.show()
if __name__ == '__main__':

    backward()