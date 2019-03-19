#encoding=utf-8
import  tensorflow as tf

lr=0.2
decay=0.9
decay_step=3

global_step=tf.Variable(0,trainable=False)

learning_rate_1=tf.train.exponential_decay(
    lr,global_step,decay_step,decay,True
)
learning_rate_2=tf.train.exponential_decay(
    lr,global_step,decay_step,decay,False
)

w=tf.Variable(tf.constant(5,dtype=tf.float32))
loss=tf.square(w+1)
train_step=tf.train.GradientDescentOptimizer(lr).minimize(loss,global_step=global_step)

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        learning_rate_val_1=sess.run(learning_rate_1)
        learning_rate_val_2 = sess.run(learning_rate_2)
        global_step_val=sess.run(global_step)
        w_val=sess.run(w)
        loss_val=sess.run(loss)
        print("After %s steps: global step is %f,w is %f, learning_rate_1 is %f,learning_rate_2 is %f,loss is %f"
              %(i,global_step_val,w_val,learning_rate_val_1,learning_rate_val_2,loss_val))