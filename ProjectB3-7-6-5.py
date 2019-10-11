#
# Project 1, starter code part b
#
import math
import tensorflow as tf
tf.disable_v2_behavior()
import numpy as np
import pylab as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#NUM_FEATURES = 6

learning_rate = 0.001
epochs = 50000
batch_size = 8
num_neuron = 10
seed = 10
reg_weight = 0.001
np.random.seed(seed)
tf.set_random_seed(seed+5)

# initialization routines for bias and weights
def init_bias(n = 1):
    return(tf.Variable(np.zeros(n), dtype=tf.float32))

def init_weights(n_in=1, n_out=1):
    W_values = tf.random.truncated_normal((n_in, n_out), stddev=1.0/math.sqrt(n_in))
    return(tf.Variable(W_values, dtype=tf.float32))

def ffn(x, NUM_FEATURES):
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
        y = tf.matmul(H, V) + c # linear

    return y, V, c, W, b

def prepare_graph(NUM_FEATURES):
    # Create the model
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    d = tf.placeholder(tf.float32, [None, 1])

    y, V, c, W, b = ffn(x, NUM_FEATURES)

    #Create the gradient descent optimizer with the given learning rate.
    mse = tf.reduce_mean(tf.square(d - y))

    regularisation = tf.nn.l2_loss(V) + tf.nn.l2_loss(W)
    loss = mse + reg_weight * regularisation

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op, loss, x, d

def main():
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

    trainXfull = np.copy(X_data[:cutoff])
    trainYfull = np.copy(Y_data[:cutoff])

    testXfull = np.copy(X_data[cutoff:])
    testYfull = np.copy(Y_data[cutoff:])

    # All 7 features
    total_train_errs = []
    total_test_errs = []

    trainX = np.copy(trainXfull)
    trainY = np.copy(trainYfull)
    testX = np.copy(testXfull)
    testY = np.copy(testYfull)
    
    train_op, loss, x, d = prepare_graph(7)

    for i, num_features in zip(range(3), [7,6,5]):

        np.random.seed(3)

        trainX = np.copy(trainXfull)
        trainY = np.copy(trainYfull)
        testX = np.copy(testXfull)
        testY = np.copy(testYfull)

        if num_features < 7:
            # remove column 2
            trainX = np.delete(trainX, 1, 1)
            testX = np.delete(testX, 1, 1)

        if num_features < 6:
            # remove column 7
            trainX = np.delete(trainX, 5, 1)
            testX = np.delete(testX, 5, 1)
        
        train_op, loss, x, d = prepare_graph(num_features)

        with tf.Session() as sess:
            tf.set_random_seed(seed+5)
            tf.global_variables_initializer().run()
            train_err = []
            test_err = []
            idx = np.arange(trainX.shape[0])

            # LEARN WEIGHTS
            for i in range(epochs):
                
                # Shuffle at every epoch
                np.random.shuffle(idx)
                trainX, trainY = trainX[idx], trainY[idx]


                for start, end in zip(range(0, len(trainX), batch_size), range(batch_size, len(trainX), batch_size)):
                    sess.run([train_op], feed_dict={x: trainX[start:end], d: trainY[start:end]})

                tr_err = sess.run([loss], feed_dict={x: trainX, d: trainY})
                train_err.append(tr_err[0])

                te_err = sess.run([loss], feed_dict={x: testX, d: testY})
                test_err.append(te_err[0])

                if i % 100 == 0:
                    print('iter %d: train error %g test error %g'%(i, train_err[i], test_err[i]))

            # Store all train/test errors
            total_train_errs.append(train_err)
            total_test_errs.append(test_err)



    # plot learning curves
    f1 = plt.figure(1)

    plt.plot(range(epochs), total_train_errs[0], label = '7 features')
    plt.plot(range(epochs), total_train_errs[1], label = '6 features')
    plt.plot(range(epochs), total_train_errs[2], label = '5 features')

    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Mean Square Error')
    plt.title('Training errors against Epochs - RFE')
    plt.ylim(0,0.015)
    plt.legend()

    f2 = plt.figure(2)

    plt.plot(range(epochs), total_test_errs[0], label = '7 features')
    plt.plot(range(epochs), total_test_errs[1], label = '6 features')
    plt.plot(range(epochs), total_test_errs[2], label = '5 features')

    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Mean Square Error')
    plt.title('Test errors against Epochs - RFE')
    plt.ylim(0,0.015)
    plt.legend()


    plt.show()

if __name__ == '__main__':
    main()