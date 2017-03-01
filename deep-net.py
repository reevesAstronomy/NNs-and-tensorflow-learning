################################################################################
# Neural network for classifying MNIST digits.
#
# Tutorial/test based on YouTube user "sentdex"'s tutorial series "Deep Learning
# with Neural Networks and Tensorflow".
#
# With n_nodes_hl1 = n_nodes_hl2 = n_nodes_hl3 = 500, batch_size=100, and
# n_epochs=10, I get a test accuracy of 94.8% or 94.9% on the MNIST dataset.
# Takes 10.87s to train using this scenario on an NVIDIA GTX 1060 6GB GPU.
################################################################################

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

# height x width
x = tf.placeholder('float', [None, 784]) #Placeholder for the data
y = tf.placeholder('float') #Placeholder for labels for the data

def neural_network_model(data):
    #Basic feed-forward neural network

    # Declare initial weight values:
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}

    # Feed-forward math operations defined below here:
    l1 = tf.add( tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'] )
    l1 = tf.nn.relu(l1) #Using rectified linear function

    l2 = tf.add( tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'] )
    l2 = tf.nn.relu(l2) #Using rectified linear function

    l3 = tf.add( tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'] )
    l3 = tf.nn.relu(l3) #Using rectified linear function

    output_layer = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output_layer

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)) #Cross entropy with logits as the cost function

    #Minimize the cost:
    optimizer = tf.train.AdamOptimizer().minimize(cost) #Note: AdamOptimizer() uses learning_rate=0.001 by default

    hm_epochs = 10 # cycles of feed-forward with backpropagation

    with tf.Session() as sess:
        # The graph's calculations are executed using sess.run():
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs): # epoch = one loop through all of the training data
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch+1, 'completed out of', hm_epochs,'loss',epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
