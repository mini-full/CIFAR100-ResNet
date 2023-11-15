import numpy as np
import matplotlib.pyplot as plt

A = np.random.rand(40, 20)
b = np.random.rand(40, 1)
x = np.random.rand(20, 1)


# Set the learning rate
lr = 1e-3
iteration = 50

loss = np.zeros(iteration)

# Gradient descent
for i in range(iteration):
    loss[i] = np.linalg.norm(np.dot(A, x) - b)
    grad = 2 * np.dot(A.T, np.dot(A, x) - b)
    x = x - lr * grad
    
# Plot the loss and save
plt.plot(loss)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.savefig('loss.png')
