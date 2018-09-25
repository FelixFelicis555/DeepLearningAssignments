import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_ = tf.placeholder(tf.float32, shape=[None,3], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[None,1], name="y-input")

Theta1 = tf.Variable(tf.random_uniform([3,3], -1, 1), name="Theta1")
Theta2 = tf.Variable(tf.random_uniform([3,1], -1, 1), name="Theta2")
Bias1 = tf.Variable(tf.zeros([3]), name="Bias1")
Bias2 = tf.Variable(tf.zeros([1]), name="Bias2")

A2 = tf.sigmoid(tf.matmul(x_, Theta1) + Bias1)
Hypothesis = tf.sigmoid(tf.matmul(A2, Theta2) + Bias2)

cost = tf.reduce_mean(( (y_ * tf.log(Hypothesis)) + 
        ((1 - y_) * tf.log(1.0 - Hypothesis)) ) * -1)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[0],[1],[1],[0]]

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
cost_hist = []
for i in range(100000):
        err, _ = sess.run([cost,train_step], feed_dict={x_: XOR_X, y_: XOR_Y})
        # Hypothesis = tf.sigmoid(tf.matmul(A2, Theta2) + Bias2)
        # cost = tf.reduce_mean(( (y_ * tf.log(Hypothesis)) + ((1 - y_) * tf.log(1.0 - Hypothesis)) ) * -1)
        cost_hist.append(err)
        if (i % 10000 == 0 ) :
            print("epoch :" + str(i) + " cost : " + str(err))
            print('Hypothesis ', sess.run(Hypothesis, feed_dict={x_: XOR_X, y_: XOR_Y}))
            

plt.figure("cost vs iterations") 
plt.plot(np.linspace(0,len(cost_hist),len(cost_hist)),np.array(cost_hist),label='cost vs iterations')  
plt.legend()
plt.show()      

#testing around the truth table values
print("Testing..)")

test_input = [
    [1.001, 1.003],
    [1.0092, 0.008],
    [0.004, 1.0024],
    [0.0033, 0.008],
    [0.951, 1.009],
    [1.101, 0.030],
    [0.064, 1.024],
    [-0.0092, -0.0072],
    [1.001, 1.005],
    [0.998, -0.0028],
    [-0.0074, 1.0104],
    [0.0045, -0.031],
    [1.0031, 0.925],
    [0.999, 0.0028],
    [0.0074, 1.00104],
    [-0.0018, -0.0029]

]
test_output = [[0.0],[1.0],[1.0],[0.0],[0.0],[1.0],[1.0],[0.0], [0.0],[1.0],[1.0],[0.0],[0.0],[1.0],[1.0],[0.0]]

test_input = np.array(test_input)
result = []
with tf.Session() as session:
    tf.global_variables_initializer().run()
    #sess.run(init)
    for i in range(int(16)):
        x_input = np.array([[float(test_input[i][0]),float(test_input[i][1])]])
        result.append(session.run(Hypothesis, feed_dict={x_: x_input}) )
        print(str(x_input )+ str(result[i]))

    session.close()

x1 = test_input[:,0]
x2 = test_input[:,1]
y = np.array(np.round(result))
print( y)
for i in range(16) :
    #print (result[1])
    if y[i] == 1.0 :
        plt.scatter(x1[i], x2[i], marker='x')
    else :
        plt.scatter(x1[i], x2[i], marker='o')

plt.xlabel('input1- axis')
plt.ylabel('input2 - axis')
plt.title('Testing - XOR')
plt.show()
