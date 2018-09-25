
import numpy as np
import sys
import matplotlib.pyplot as plt

def step(self, y):
    return 1 if y >= 0 else 0

def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation > 0.001 else 0.0
 

def train_weights(train, alpha, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0.0
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			sum_error += error**2
			weights[0] = weights[0] + alpha * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] +alpha * error * row[i]
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, alpha, sum_error))
	return weights
 
    
T, F = 1.0, 0.0

training_input = [
    [T, T, T],
    [T, F, T],
    [F, T, T],
    [F, F, F],
]


epoch, max_epochs = 0, 40
cost_hist = []
alpha = 0.01

weights = train_weights(training_input, alpha, max_epochs)
print(weights)


cf = weights    #coefficients of line

plt.figure("linear hyperplane") 
x = np.linspace(-4.,4.)
plt.plot(x,(-cf[1]*x  - cf[0])/cf[2])
data = np.array([
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0]
])
x, y = data.T
plt.scatter(x,y)
plt.show()


#testing around the truth table values
print("Testing... evaluating accuracy)")
bias = 1.0
test_input = [
    [bias , 1.001, 1.003 ],
    [bias, 1.0092, 0.008 ],
    [bias, 0.004, 1.0024],
    [bias, 0.0033, 0.008],
    [bias, 0.0051, 1.009],
    [bias, 1.101, 0.030],
    [bias, 0.064, 1.024],
    [bias, 0.0092, 0.0072],
    [bias, 1.001, 1.005],
    [bias, .998, 1.0028],
    [bias, .0074, 1.0104],
    [bias, .045, .31],
    [bias, 1.31, .925],
    [bias, .9, 1.0028],
    [bias, 1.0074, .104],
    [bias, .18, .29]

]
test_output = [
    [T],
    [T],
    [T],
    [F],
    [T],
    [T],
    [T],
    [F],
    [T],
    [T],
    [T],
    [F],
    [T],
    [T],
    [T],
    [F]
]

out = np.matmul(test_input,cf )
print (out)
cnt = 0
for i in range(16) :
    if out[i] > 0 :
    	out[i] = 1.0
    else :
        out[i] = 0.0
    if out[i] == test_output[i]:
        cnt += 1
accuracy = ( cnt / 16 )*100
print ("accuracy is" , accuracy)


