import os
import pandas as pd
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
display_step = 1

# dog vs. cat classification problem
# features: height, weight, lenght
# labels: 0 -> cat, 1 -> dog

n_features = 3
n_classes = 2

data = pd.read_csv('data/tf/tf_demo_data.csv').values
lbl = pd.read_csv('data/tf/tf_demo_labels.csv').values

training_epochs = 15
# number of neurons
n_neurons = 32

###
# Keep in mind, the "neural network" is just a matrix multiplication and a vector addition
# let's declare all the necessary components
###

# tensorflow requires us to declare two placeholders, tensors that contain input and output but don't perform any operation
x = tf.placeholder('float', [None, n_features])
y = tf.placeholder('float', [None, n_classes])
# the weights matrix
W1 = tf.Variable(tf.random_normal([n_features, n_neurons]))
# the bias vector
b1 = tf.Variable(tf.random_normal([n_neurons]))

# the second matrix and vector
W2 = tf.Variable(tf.random_normal([n_neurons, n_classes]))
b2 = tf.Variable(tf.random_normal([n_classes]))

# the two "layers" of our neural network, with a RELU in between
layer1 = tf.add(tf.matmul(x, W1), b1)
relu_layer = tf.nn.relu(layer1)
layer2 = tf.add(tf.matmul(relu_layer, W2), b2)

# the loss function is performing softmax internally
loss_op = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=layer2, labels=y)
)
# Stocastic Gradient Descent to potimize the model
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

# we want our optimizer (SGD) to minimize the loss (CrossEntropy)
train_op = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()

# starting the tensorflow session
with tf.Session() as sess:
    sess.run(init)

    # we are training the model using batches
    batch_size = 100
    n_batches = data.shape[0]//batch_size
    # for each training epoch, we optimize the model using the whole dataset
    for epoch in range(training_epochs):
        avg_cost = 0
        for i in range(n_batches):
            batch = data[i*batch_size:(i+1)*batch_size]
            batch_lbl = lbl[i*batch_size:(i+1)*batch_size]
            _, c = sess.run(
                [train_op, loss_op],
                feed_dict = {x: batch, y: batch_lbl}
            )
            avg_cost += c / n_batches
        if epoch % display_step == 0:
            print('Epoch: %d, Cost:%f' % ((epoch+1), avg_cost))
    print('Training is over!')

    # after the training, let's try to get some predictions
    print('Predicting samples [28, 18, 60] and [15, 8, 40]; expected class: 1 and 0')
    sample = [[28, 18, 60], [15, 8, 40]]
    # in order to get a prediction:
    # the tensor x (the placeholder for the input) contains the samples
    # tensorflow is running our model and is applying the softmax function
    print(sess.run([tf.nn.softmax(layer2)], feed_dict={x: sample}))
