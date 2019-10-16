#
# Project 1, starter code part b
#
import math
import tensorflow as tf
tf.disable_v2_behavior()
import numpy as np
import pylab as plt
import multiprocessing as mp

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

NUM_FEATURES = 5

learning_rate = 0.001
epochs = 100000
batch_size = 8
num_neuron = 10
num_neuron_4_layer = 50
seed = 10
reg_weight = 0.001
np.random.seed(seed)
tf.set_random_seed(seed+5)
DROPOUT_KEEP_RATE = 0.8
NN_sizes = [3,3,4,4,5,5]
dropouts = [0,1,0,1,0,1] # 0 = no dropout 1 = dropout
legend = ["3 layer", "3 layer w/ dropout", "4 layer", "4 layer w/ dropout", "5 layer", "5 layer w/ dropout"]

def init_data():
    #read and divide data into test and train sets
    admit_data = np.genfromtxt('admission_predict.csv', delimiter= ',')
    X_data, Y_data = admit_data[1:,1:8], admit_data[1:,-1]
    Y_data = Y_data.reshape(Y_data.shape[0], 1)

    idx = np.arange(X_data.shape[0])
    np.random.shuffle(idx)
    X_data, Y_data = X_data[idx], Y_data[idx]

    X_data = (X_data- np.mean(X_data, axis=0))/ np.std(X_data, axis=0)

    #split into train and test
    cutoff = math.floor(0.7 * len(X_data))

    trainX = np.copy(X_data[:cutoff])
    trainY = np.copy(Y_data[:cutoff])

    testX = np.copy(X_data[cutoff:])
    testY = np.copy(Y_data[cutoff:])

    return trainX, trainY, testX, testY

def deleteRows(dataset):
    #delete row 2
    dataset = np.delete(dataset, 1, 1) #1 because offset by 1
    #delete row 7
    dataset = np.delete(dataset, 5, 1) #5 because offset by 1 - 1
    return dataset

trainXfull, trainYfull, testXfull, testYfull = init_data()
trainXfull = deleteRows(trainXfull)
testXfull = deleteRows(testXfull)

# initialization routines for bias and weights
def init_bias(n = 1):
    return(tf.Variable(np.zeros(n), dtype=tf.float32))

def init_weights(n_in=1, n_out=1):
    W_values = tf.random.truncated_normal((n_in, n_out), stddev=1.0/math.sqrt(n_in), seed=seed)
    return(tf.Variable(W_values, dtype=tf.float32))

def threelayerffn(x, dropout_rate):
    with tf.name_scope('hidden'):

        # Hidden Layer
        W = init_weights(NUM_FEATURES, num_neuron)
        b = init_bias(num_neuron)
        Z = tf.matmul(x, W) + b
        H = tf.nn.relu(Z)

    with tf.name_scope('linear'):

        # Output Layer
        V = init_weights(num_neuron, 1)
        c = init_bias(1)
        drop_out_h = tf.nn.dropout(H, dropout_rate)
        y = tf.matmul(drop_out_h, V) + c # linear

    return y, V, c, W, b

def fourlayerffn(x, dropout_rate):
    with tf.name_scope('hidden'):

        # Hidden Layer
        W = init_weights(NUM_FEATURES, num_neuron_4_layer)
        b = init_bias(num_neuron_4_layer)
        Z = tf.matmul(x, W) + b
        H = tf.nn.relu(Z)
        drop_out_h = tf.nn.dropout(H, dropout_rate)

        W1 = init_weights(num_neuron_4_layer, num_neuron_4_layer)
        b1 = init_bias(num_neuron_4_layer)
        Z1 = tf.matmul(drop_out_h, W1) + b1
        H1 = tf.nn.relu(Z1)
        drop_out_h1 = tf.nn.dropout(H1, dropout_rate)

    with tf.name_scope('linear'):

        # Output Layer
        V = init_weights(num_neuron_4_layer, 1)
        c = init_bias(1)
        
        y = tf.matmul(drop_out_h1, V) + c # linear

    return y, V, c, W, b

def fivelayerffn(x, dropout_rate):
    with tf.name_scope('hidden'):

        # Hidden Layer
        W = init_weights(NUM_FEATURES, num_neuron_4_layer)
        b = init_bias(num_neuron_4_layer)
        Z = tf.matmul(x, W) + b
        H = tf.nn.relu(Z)
        drop_out_h = tf.nn.dropout(H, dropout_rate)

        W1 = init_weights(num_neuron_4_layer, num_neuron_4_layer)
        b1 = init_bias(num_neuron_4_layer)
        Z1 = tf.matmul(drop_out_h, W1) + b1
        H1 = tf.nn.relu(Z1)
        drop_out_h1 = tf.nn.dropout(H1, dropout_rate)

        W2 = init_weights(num_neuron_4_layer, num_neuron_4_layer)
        b2 = init_bias(num_neuron_4_layer)
        Z2 = tf.matmul(drop_out_h1, W2) + b2
        H2 = tf.nn.relu(Z2)
        drop_out_h2 = tf.nn.dropout(H2, dropout_rate)

    with tf.name_scope('linear'):

        # Output Layer
        V = init_weights(num_neuron_4_layer, 1)
        c = init_bias(1)
        
        y = tf.matmul(drop_out_h2, V) + c # linear

    return y, V, c, W, b


def runNeuralNetworks(params):

    # make local copy of dataset to not affect other sets
    trainX = np.copy(trainXfull)
    trainY = np.copy(trainYfull)
    testX = np.copy(testXfull)
    testY = np.copy(testYfull)

    np.random.seed(seed+6)
    tf.set_random_seed(seed+5)
    nn_size, isDropout = params[0], params[1]

    if isDropout:
        dropoutrate = DROPOUT_KEEP_RATE # set in the parameters above
    else:
        dropoutrate = 1 # no dropouts

    # Create the model
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    d = tf.placeholder(tf.float32, [None, 1])
    dropout = tf.placeholder(tf.float32)

    if nn_size == 3:
        ffn = threelayerffn
    elif nn_size == 4:
        ffn = fourlayerffn
    elif nn_size == 5:
        ffn = fivelayerffn

    y, V, c, W, b = ffn(x, dropout)

    #Create the gradient descent optimizer with the given learning rate.
    mse = tf.reduce_mean(tf.square(d - y))

    regularisation = tf.nn.l2_loss(V) + tf.nn.l2_loss(W)
    loss = mse + reg_weight * regularisation

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    train_err = []
    test_err = []

    with tf.Session() as sess:
        tf.set_random_seed(5)
        tf.global_variables_initializer().run()
        idx = np.arange(trainX.shape[0])

        # LEARN WEIGHTS
        for i in range(epochs):

            # Shuffle at every epoch

            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, len(trainX), batch_size), range(batch_size, len(trainX), batch_size)):
                sess.run([train_op], feed_dict={x: trainX[start:end], d: trainY[start:end], dropout: dropoutrate})

            tr_err = sess.run([loss], feed_dict={x: trainX, d: trainY, dropout:1})
            train_err.append(tr_err[0])

            te_err = sess.run([loss], feed_dict={x: testX, d: testY, dropout:1})
            test_err.append(te_err[0])

            #if i % 100 == 0:
                #print('iter %d: train error %g test error %g'%(i, train_err[i], test_err[i]))
            if i % (epochs/10) == 0: # every 10% progress, print out the train/test error
                outputstr = f"epoch {i} - {nn_size} layer NN - dropout={isDropout} - train error={train_err[i]} test error={test_err[i]}"
                print(outputstr)
            
        print(f"complete - {nn_size} layer NN - dropout={isDropout} - train error={train_err[i]} test error={test_err[i]}")

    return [train_err, test_err]


def main():

    # Split to multiple threads
    no_threads = mp.cpu_count()
    p = mp.Pool(processes=no_threads)
    results = p.map(runNeuralNetworks, zip(NN_sizes, dropouts))

    # Parse results
    train_err, test_err = zip(*results)

    f1 = plt.figure(1)

    for err, label in zip(train_err, legend):
        plt.plot(range(epochs), err, label = label)

    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Mean Square Error')
    plt.title('Training errors against Epochs - (5 vs 6 vs 7 layer) with/without dropouts')
    plt.ylim(0,0.010)
    plt.legend()

    f2 = plt.figure(2)

    for err, label in zip(test_err, legend):
        plt.plot(range(epochs), err, label = label)

    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Mean Square Error')
    plt.title('Test errors against Epochs - (5 vs 6 vs 7 layer) with/without dropouts')
    plt.ylim(0,0.010)
    plt.legend()


    plt.show()

if __name__ == '__main__':
    main()