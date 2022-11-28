import numpy as np
from time import time
import matplotlib.pyplot as plt
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

# Parameters
learning_rate = 0.01
training_epochs = 1000
num_labels = 3
batch_size = 100

# Learn data & Test data
test_count = 10
learn_count = 100
all_count = test_count + learn_count

x1_label0 = np.random.normal(1, 1, (all_count, 1))
x2_label0 = np.random.normal(1, 1, (all_count, 1))
x1_label1 = np.random.normal(5, 1, (all_count, 1))
x2_label1 = np.random.normal(4, 1, (all_count, 1))
x1_label2 = np.random.normal(8, 1, (all_count, 1))
x2_label2 = np.random.normal(0, 1, (all_count, 1))

xs_label0 = np.hstack((x1_label0, x2_label0))
xs_label1 = np.hstack((x1_label1, x2_label1))
xs_label2 = np.hstack((x1_label2, x2_label2))
xs = np.vstack((xs_label0, xs_label1, xs_label2))

labels = np.matrix([[1., 0., 0.]] * xs_label0.shape[0] +
                   [[0., 1., 0.]] * xs_label1.shape[0] +
                   [[0., 0., 1.]] * xs_label2.shape[0])

arr = np.arange(xs.shape[0])
np.random.shuffle(arr)
learn_arr, test_arr = arr[:learn_count*num_labels], arr[learn_count*num_labels:]
xs, test_xs = xs[learn_arr, :], xs[test_arr, :]
labels, test_labels = labels[learn_arr, :], labels[test_arr, :]

train_size, num_features = xs.shape


# TensorFlow
start = time()

X = tf.placeholder(tf.float32, shape=(None, num_features), name='inputs')
Y = tf.placeholder(tf.float32, shape=(None, num_labels), name='outputs')

w = tf.Variable(tf.zeros([num_features, num_labels]))
b = tf.Variable(tf.zeros([num_labels]))
y_model = tf.nn.softmax(tf.matmul(X, w) + b)  # tf.sm(X @ w + b) --- matmul перемножение матриц

cost = tf.reduce_sum(Y * -tf.log(y_model))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for step in range(training_epochs * train_size // batch_size):
        offset = (step * batch_size) % train_size
        batch_xs = xs[offset:(offset + batch_size), :]
        batch_labels = labels[offset:(offset + batch_size), :]

        curr_cost, _ = sess.run([cost, train_op], feed_dict={X: batch_xs, Y: batch_labels})
        print(f"StepN: {step + 1}\tCost: {curr_cost :.5f}")

    y, x = [0., 1., 0.], [5.99, 3.25]

    w_val = sess.run(w)
    b_val = sess.run(b)
    accuracy_test = accuracy.eval({X: test_xs, Y: test_labels})

print(f"W: {w_val}\nB: {b_val}\n"
      f"Accuracy_test: {accuracy_test}\n"
      f"Time: {time() - start :.2f}")

# Show data
plt.scatter(x1_label0, x2_label0, c='r', marker='o', s=60)
plt.scatter(x1_label1, x2_label1, c='g', marker='x', s=60)
plt.scatter(x1_label2, x2_label2, c='b', marker='_', s=60)
plt.show()
