
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#one hot to provide binarization
mnist_data=input_data.read_data_sets('MNIST_data',one_hot=True)
# y = Wx + b
# Input to the graph, takes in any number of images (784 element pixel arrays)
x_input = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x_input')
W = tf.Variable(initial_value=tf.zeros(shape=[784, 10]), name='W')
# Biases = weights * inputs
b = tf.Variable(initial_value=tf.zeros(shape=[10]), name='b')
y_actual = tf.add(x=tf.matmul(a=x_input, b=W, name='matmul'), y=b, name='y_actual')
y_expected = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y_expected')
# Cross entropy loss function because output is a list of possibilities (% certainty of the correct answer)
cross_entropy_loss = tf.reduce_mean(
input_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=y_expected, logits=y_actual),
name='cross_entropy_loss')
# Classic gradient descent optimizer aims to minimize the difference between expected and actual values (loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5, name='optimizer')
train_step = optimizer.minimize(loss=cross_entropy_loss, name='train_step')
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
for _ in range(1000):
    batch =mnist_data.train.next_batch(100)
    train_step.run(feed_dict={x_input: batch[0], y_expected: batch[1]})
# Measure accuracy by comparing the predicted values to the correct values and calculating how many of them match
correct_prediction = tf.equal(x=tf.argmax(y_actual, 1), y=tf.argmax(y_expected, 1))
accuracy = tf.reduce_mean(tf.cast(x=correct_prediction, dtype=tf.float32))
print(100*accuracy.eval(feed_dict={x_input: mnist_data.test.images, y_expected: mnist_data.test.labels}))
# Test a prediction on a single image
print(session.run(fetches=y_actual, feed_dict={x_input: [mnist_data.test.images[0]]}))
session.close()

