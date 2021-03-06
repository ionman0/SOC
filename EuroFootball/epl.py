import pandas as pd
import numpy as np
import tensorflow as tf


# Used to map results
def resultEncoder(hg, ag):
    if hg > ag:
        return np.array([1, 0, 0])
    elif ag > hg:
        return np.array([0, 0, 1])
    else:
        return np.array([0, 1, 0])


# Read datasets
testset = pd.read_csv('testset.csv')
trainset = pd.read_csv('trainset.csv')

# drop unrequired shit
testset = testset.dropna(1, 'any')
trainset = trainset.dropna(1, 'any')
testset = testset.drop(['season', 'id'], 1)
trainset = trainset.drop(['season', 'id'], 1)

trainResults = np.empty([2660, 3])
testResults = np.empty([380, 3])

# Map the results using resultEncoder
for index, row in trainset.iterrows():
    hg = row['home_team_goal']
    ag = row['away_team_goal']
    trainResults[index] = resultEncoder(hg, ag)

for index, row in testset.iterrows():
    hg = row['home_team_goal']
    ag = row['away_team_goal']
    testResults[index] = resultEncoder(hg, ag)

x = tf.placeholder(tf.float32, [None, 19])
W = tf.Variable(tf.zeros([19, 3]))
b = tf.Variable(tf.zeros(3))

y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 3])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()

# Train
tf.global_variables_initializer().run()
for i in range(10000):
    batch_xs = trainset.sample(500)
    batch_ys = trainResults[batch_xs.index]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: testset, y_: testResults}))
