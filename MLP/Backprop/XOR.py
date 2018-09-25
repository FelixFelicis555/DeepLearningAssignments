import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt


training_input = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
training_output = [[0],[1],[1],[0],[1],[0],[0],[1]]

inputs = tf.placeholder(tf.float32, shape=[None,3], name="x-input")
targets = tf.placeholder(tf.float32, shape=[None,1], name="y-input")

Theta1 = tf.Variable(tf.random_uniform([3,3], -1, 1), name="Theta1")
Theta2 = tf.Variable(tf.random_uniform([3,1], -1, 1), name="Theta2")
Bias1 = tf.Variable(tf.zeros([3]), name="Bias1")
Bias2 = tf.Variable(tf.zeros([1]), name="Bias2")

A2 = tf.sigmoid(tf.matmul(inputs, Theta1) + Bias1)
Hypothesis = tf.sigmoid(tf.matmul(A2, Theta2) + Bias2)

cost = tf.reduce_mean(( (targets * tf.log(Hypothesis)) + 
        ((1 - targets) * tf.log(1.0 - Hypothesis)) ) * -1)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
cost_hist = []
for i in range(130000): 
        err, _ = sess.run([cost,train_step], feed_dict={inputs: training_input, targets: training_output})
      
        cost_hist.append(err)
        if (i % 10000 == 0 ) :
            print("epoch :" + str(i) + " cost : " + str(err))
            print('Hypothesis ', sess.run(Hypothesis, feed_dict={inputs : training_input, targets : training_output}))
            

plt.figure("cost vs iterations") 
plt.plot(np.linspace(0,len(cost_hist),len(cost_hist)),np.array(cost_hist),label='cost vs iterations')  
plt.legend()
plt.show()      

#testing around the truth table values
print("Testing..)")

test_input = [[0.200,0.300,0.022],[0.33,-0.44,1.233],[0.1,1.9,0.7],[0.189,1.7,1.1],[1.1,0.2,0.2],[1.4,0.01,1.002],[1.33,1.029,0.02],[1.01,1.20,1.213]]
test_output = [[0],[1],[1],[0],[1],[0],[0],[1]]

test_input = np.array(test_input)
result = []
with tf.Session() as session:
    tf.global_variables_initializer().run()
    #sess.run(init)
    for i in range(int(8)):
        x_input = np.array(test_input[i])
        result.append(session.run(Hypothesis, feed_dict={inputs : test_input, targets : test_output}))
        print(str(x_input )+ str(result[i]))

    session.close()
sys.exit(0)
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
