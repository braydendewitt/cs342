import numpy as np
# Given values
x_i = np.array([np.log(2), np.log(3)]).reshape(-1, 1)  # Column vector
y_i = 2  # Class 2 in one-hot encoding is [0, 1, 0]
W = np.array([[1, 0, 0], [0, 1, 0]])
b = np.array([[0], [0], [0]])  # Column vector for bias

# Number of classes, C
C = W.shape[1]

# Compute the logits o = W^T x_i + b
o = np.dot(W.T, x_i) + b
print(o)

# Compute the softmax probabilities for each class
exp_o = np.exp(o)
sum_exp_o = np.sum(exp_o)
s_o = exp_o / sum_exp_o
print(s_o)

# Indicator vector for the true class y_i
one_hot_y_i = np.zeros((C, 1))
one_hot_y_i[y_i - 1] = 1  # Subtract 1 because class indices are 0-based in Python

# Compute the gradients
grad_W = np.dot(x_i, (s_o - one_hot_y_i).T)
grad_b = s_o - one_hot_y_i

# Round the gradients to three decimal places
grad_W_rounded = np.round(grad_W, 3)
grad_b_rounded = np.round(grad_b, 3)

grad_W_rounded, grad_b_rounded
print(grad_W_rounded)
print(grad_b_rounded)