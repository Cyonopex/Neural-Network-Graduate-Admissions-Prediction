#
# Project 1, starter code part b
#
import math
import tensorflow as tf
tf.disable_v2_behavior()
import numpy as np
import pylab as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

NUM_FEATURES = 7

learning_rate = 0.001
epochs = 80000
batch_size = 8
num_neuron = 10
seed = 10
reg_weight = 0.001
stopping_epoch = 24000
np.random.seed(seed)
tf.set_random_seed(seed)

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

# initialization routines for bias and weights
def init_bias(n = 1):
    return(tf.Variable(np.zeros(n), dtype=tf.float32))

def init_weights(n_in=1, n_out=1):
    W_values = tf.random.truncated_normal((n_in, n_out), stddev=1.0/math.sqrt(n_in))
    return(tf.Variable(W_values, dtype=tf.float32))

# Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
d = tf.placeholder(tf.float32, [None, 1])

# Init variables
V = init_weights(num_neuron, 1)
c = init_bias(1)
W = init_weights(NUM_FEATURES, num_neuron)
b = init_bias(num_neuron)

# Build graph

# Hidden Layer
Z = tf.matmul(x, W) + b
H = tf.nn.relu(Z)

# Output Layer
y = tf.matmul(H, V) + c # linear

#Create the gradient descent optimizer with the given learning rate.
mse = tf.reduce_mean(tf.square(d - y))

regularisation = tf.nn.l2_loss(V) + tf.nn.l2_loss(W)
loss = mse + reg_weight * regularisation

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_err = []
    log_train_err = []
    test_err = []
    log_test_err = []
    idx = np.arange(trainX.shape[0])
    V_, c_, W_, b_ = 0,0,0,0

    # LEARN WEIGHTS
    for i in range(epochs):
        
        # Shuffle at every epoch
        np.random.shuffle(idx)
        trainX, trainY = trainX[idx], trainY[idx]

        for start, end in zip(range(0, len(trainX), batch_size), range(batch_size, len(trainX), batch_size)):
            train_op.run(feed_dict={x: trainX[start:end], d: trainY[start:end]})
        
        if i == stopping_epoch: # at the selected epoch, store all weights for future computation
            V_, c_, W_, b_ = sess.run([V, c, W, b])

        tr_err = loss.eval(feed_dict={x: trainX, d: trainY})
        train_err.append(tr_err)
        log_train_err.append(math.log10(tr_err))

        te_err = loss.eval(feed_dict={x: testX, d: testY})
        test_err.append(te_err)
        log_test_err.append(math.log10(te_err))

        if i % 100 == 0:
            print('iter %d: train error %g test error %g'%(i, train_err[i], test_err[i]))

    # APPLY SAMPLE DATA

    # load weights from selected epoch
    V.load(V_, sess)
    W.load(W_, sess)
    b.load(b_, sess)
    c.load(c_, sess)

    # Shuffle test samples and test 50 of them
    idxtest = np.arange(testX.shape[0])
    np.random.shuffle(idxtest)
    testX, testY = testX[idxtest], testY[idxtest]
    sub_testX = testX[:50]
    sub_testY = testY[:50]

    #print(sub_testX)
    #print(sub_testY)

    idxsort = np.argsort([i[0] for i in sub_testY])
    sub_testX, sub_testY = sub_testX[idxsort], sub_testY[idxsort]

    y = sess.run([y], feed_dict={x: sub_testX})

    #print(type(y))


# plot learning curves
#with plt.xkcd():
f1 = plt.figure(1)
plt.plot(range(epochs), train_err, label = 'train error')
plt.plot(range(epochs), test_err, label = 'test error')
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Mean Square Error')
plt.title('Training and Testing errors against Epochs')
plt.ylim(0,0.03)
plt.legend()

f2 = plt.figure(2)
y1 = [i[0] for i in y[0]]
d1 = [i[0] for i in sub_testY]

plt.scatter(range(50), y1, label = "Predicted Values")
plt.scatter(range(50), d1, label = "Actual values")
plt.title('Scatterplot of Predicted vs Target values')
plt.xlabel('50 random data samples, sorted')
plt.ylabel('Probability of admission')
plt.legend()

plt.show()
