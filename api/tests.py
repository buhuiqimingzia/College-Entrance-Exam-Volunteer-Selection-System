import tensorflow.compat.v1 as tf
import math

tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability as tfp

'''
data=np.loadtxt('/root/JUPYTER/shanfangfang/L1.txt',dtype='float',delimiter=',')
data1=data[:,0]
data2=data[:,1]
'''
x_train1 = np.linspace(0, 1, 1000, endpoint=True)  # 生成[0,2]区间100个点
x_train2 = np.linspace(1, 40, 1000, endpoint=True)
x_t1 = np.zeros((len(x_train1), 1))
x_t2 = np.zeros((len(x_train2), 1))
for i in range(len(x_train1)):
    x_t1[i] = x_train1[i]
for i in range(len(x_train2)):
    x_t2[i] = x_train2[i]
x_t = np.concatenate((x_t1, x_t2))

x1 = tf.placeholder("float", [None, 1])  # 一次传入100个点[100,1]
nv = tf.sqrt(6 / (1 + 100))
W = tf.Variable(tf.random_uniform([1, 100], -nv, nv))
b = tf.Variable(tf.random_normal([100]))
y1 = tf.nn.sigmoid(tf.matmul(x1, W) + b)  # sigmoid激活函数
nu = tf.sqrt(6 / (20 + 100))
W1 = tf.Variable(tf.random_uniform([100, 20], -nu, nu))
b1 = tf.Variable(tf.random_normal([20]))
y2 = tf.nn.sigmoid(tf.matmul(y1, W1) + b1)
nx = tf.sqrt(6 / (20 + 1))
W2 = tf.Variable(tf.random_uniform([20, 1], -nx, nx))
b2 = tf.Variable(tf.random_normal([1]))
sml = 1e-2
y3 = tf.matmul(y2, W2) + b2
y = y3 * tf.exp(-2 / (x1 + sml))
dif1 = tf.gradients(y, x1)
dif2 = tf.gradients(dif1, x1)
dif1 = tf.reshape(dif1, [2000, 1], "float")
dif2 = tf.reshape(dif2, [2000, 1], "float")

rhog = 1.2
rhol = 710.0
drho = rhol - rhog
miug = 0.0000186
miul = 0.00022
R = 0.0004
sigma = 0.0166
g = 9.8
L = 0.016
theta = 10 * math.pi / 180
z0 = 2.0 * sigma * math.cos(theta) / drho / g / R
t0 = 8.0 * miul * z0 / drho / g / R / R
A = rhog * L / t0 / t0 / drho / g
B = z0 / t0 / t0 / g
C = 1 - miug / miul
D = miug * L / miul / z0

'''
A=0.22138567
B=11.49418209
C=B
D=0.980802456
E=0.084772727
'''
# t_loss=(10*(A*diff+B*y*diff+C*dif*dif-10+(10-E)*y*dif+D*dif+y))**2#常微分方程F的平方
# t_loss=(A*dif2+B*y*dif2+C*dif1*dif1-1+(1-E)*y*dif1+D*dif1+y)#常微分方程F的平方
t_loss = (A * dif2 + B * y * dif2 + B * dif1 * dif1 + C * y * dif1 + D * dif1 + y - 1)
h = tf.zeros((2000, 1))
hubers = tf.losses.huber_loss(h, t_loss)
# loss = tf.reduce_sum(hubers)
loss = tf.reduce_sum(hubers) + ((dif1[0]) ** 2 + (y[0]) ** 2)
# loss = tf.reduce_mean(t_loss)#每点F平方求和后取平均再加上边界条件
train_step1 = tf.train.AdamOptimizer(0.02).minimize(loss)  # Adam优化器训练网络参数
train_step2 = tf.train.AdamOptimizer(0.002).minimize(loss)  # Adam优化器训练网络参数
train_step3 = tf.train.AdamOptimizer(0.001).minimize(loss)  # Adam优化器训练网络参数
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
# for i in range(50000):#训练50000次
for i in range(20000):
    # if i<5000:
    if i < 5000:
        sess.run(train_step1, feed_dict={x1: x_t})
        # if i%200 == 0:
        if i % 200 == 0:
            total_loss = sess.run(loss, feed_dict={x1: x_t})
            # print("loss={}".format(total_loss))
            # print("loss=%.8f" % total_loss)
            print(f"loss={total_loss}")

    # if i>=5000 and i<30000:
    if i >= 5000 and i < 10000:
        sess.run(train_step2, feed_dict={x1: x_t})
        if i % 200 == 0:
            # if i%100 == 0:
            total_loss = sess.run(loss, feed_dict={x1: x_t})
            # print("loss={}".format(total_loss))
            # print("loss=%.8f" % total_loss)
            print(f"loss={total_loss}")

    # if i>=30000 and i<=50000:
    if i >= 10000 and i <= 20000:
        sess.run(train_step3, feed_dict={x1: x_t})
        if i % 200 == 0:
            # if i%100 == 0:
            total_loss = sess.run(loss, feed_dict={x1: x_t})
            # print("loss={}".format(total_loss))
            # print("loss=%.8f" % total_loss)
            print(f"loss={total_loss}")  # 语法f'{变量}'

gradients = tf.gradients(loss, x1)[0]
initial_position = tf.Variable(x_t.astype(np.float32), name="initial_position")

# Define the L-BFGS optimizer

lbfgs_optimizer = tfp.optimizer.lbfgs_minimize(
    value_and_gradients_function=lambda x: sess.run([loss, gradients], feed_dict={x1: x}),
    initial_position=x_t.astype(np.float32), max_iterations=100
)

# Run the L-BFGS optimization
sess.run(lbfgs_optimizer)

saver = tf.train.Saver(max_to_keep=1)  # 保存模型，训练一次后可以将训练过程注释掉
# saver.save(sess,'ckpt1d/nn.ckpt',global_step=50000)
saver.save(sess, 'ckpt1d/nn.ckpt', global_step=20000)
saver = tf.train.Saver(max_to_keep=1)
# model_file="ckpt1d/nn.ckpt-50000"
model_file = "ckpt1d/nn.ckpt-20000"
saver.restore(sess, model_file)
output = sess.run(y, feed_dict={x1: x_t})
y_output = x_t.copy()
for i in range(len(x_t)):
    y_output[i] = output[i]
fig = plt.figure("预测曲线与实际曲线")
plt.plot(x_t, y_output, color='blue', label='DNNs')
plt.legend()
'''
plt.scatter(data1,data2,color='red',label='experiment')
plt.legend()
'''
plt.xlabel("t")
plt.ylabel("h")
plt.show()
x_t = np.reshape(x_t, [2000, 1])
y_output = np.reshape(y_output, [2000, 1])
uv = np.concatenate((x_t, y_output), 1)
# np.savetxt('/root/JUPYTER/shanfangfang/L1.dat',uv,delimiter=' ')
np.savetxt('D:/AAMM_revision/L1.dat', uv, delimiter=' ')