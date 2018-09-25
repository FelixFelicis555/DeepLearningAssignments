import tensorflow as tf 
import numpy as np
import sys
import matplotlib.pyplot as plt

T, F = 1.0, -1.0  

bias = 1.0
training_input = [
    [T, T, bias],
    [T, F, bias],
    [F, T, bias],
    [F, F, bias],
]
training_output = [
    [F],
    [F],
    [F],
    [T],
]
inputs=tf.placeholder('float',[None,3],name='Input')
targets=tf.placeholder('float',name='Target')

w = tf.Variable(tf.random_normal([3, 1]), dtype=tf.float32)

output1 = tf.matmul(training_input, w)
output = tf.subtract(tf.multiply(tf.to_float(tf.greater(output1, 0)),2),1)

error = tf.subtract(training_output, output)

mse = tf.reduce_mean(tf.square(error))

delta = tf.matmul(training_input, error, transpose_a=True)
train = tf.assign(w, tf.add(w, delta))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

err = 1
target = 0
epoch, max_epochs = 0, 40
cost_hist = []
while err > target and epoch < max_epochs:
    epoch += 1
    err, _ = sess.run([mse, train])
    cost_hist.append(err)
    print('epoch:', epoch, 'mse:', err)

print(sess.run(w))


cf = sess.run(w)

plt.figure("linear hyperplane") 
x = np.linspace(-4.,4.)
plt.plot(x,(-cf[0]*x  - cf[2])/cf[1])
data = np.array([
    [1, -1],
    [-1, 1],
    [1, 1],
    [-1, -1]
])
x, y = data.T
plt.scatter(x,y)
plt.show()

plt.figure("cost vs iterations") 
plt.plot(np.linspace(0,len(cost_hist),len(cost_hist)),np.array(cost_hist),label='cost vs iterations')  
plt.legend()
plt.show()


#testing around the truth table values
print("Testing... evaluating accuracy)")

test_input = [
    [1.001, 1.003, bias],
    [1.0092, 0.008, bias],
    [0.004, 1.0024, bias],
    [0.0033, 0.008, bias],
    [0.951, 1.009, bias],
    [1.101, 0.030, bias],
    [0.064, 1.024, bias],
    [-0.0092, -0.0072, bias],
    [1.001, 1.005, bias],
    [0.998, -0.0028, bias],
    [-0.0074, 1.0104, bias],
    [0.0045, -0.031, bias],
    [1.0031, 0.925, bias],
    [0.999, 0.0028, bias],
    [0.0074, 1.00104, bias],
    [-0.0018, -0.0029, bias]

]
test_output = [
    [F],
    [F],
    [F],
    [T],
    [F],
    [F],
    [F],
    [T],
    [F],
    [F],
    [F],
    [T],
    [F],
    [F],
    [F],
    [T]
]
#testing around the truth table values
print("Testing... evaluating accuracy)")

test_input = [
    [1.001, 1.003, bias],
    [1.0092, -1.008, bias],
    [-1.004, 1.0024, bias],
    [-1.0033, -1.008, bias],
    [-1.0051, 1.009, bias],
    [1.101, -1.030, bias],
    [-1.064, 1.024, bias],
    [-1.0092, -1.0072, bias],
    [1.001, 1.005, bias],
    [-1.998, 1.0028, bias],
    [-1.0074, 1.0104, bias],
    [-1.045, -1.31, bias],
    [1.31, -1.925, bias],
    [-1.9, 1.0028, bias],
    [1.0074, -1.104, bias],
    [-1.18, -1.29, bias]

]
test_output = [
    [F],
    [F],
    [F],
    [T],
    [F],
    [F],
    [F],
    [T],
    [F],
    [F],
    [F],
    [T],
    [F],
    [F],
    [F],
    [T]
]


out = np.matmul(test_input,cf )
print (out)
cnt = 0
for i in range(16) :
    if out[i] > 0 :
    	out[i] = -1.0
    else :
        out[i] = 1.0
    if out[i] == test_output[i]:
        cnt += 1
accuracy = ( cnt / 16 )*100
print ("accuracy is" , accuracy)
