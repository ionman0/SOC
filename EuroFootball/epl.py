import pandas as pd
import numpy as np
import tensorflow as tf

def resultEncoder(hg, ag):
	if	 hg > ag: return np.array([1,0,0])
	elif ag > hg: return np.array([0,0,1])
	else:		  return np.array([0,1,0])

testset = pd.read_csv('testset.csv')
trainset = pd.read_csv('trainset.csv')
testset = testset.dropna(1, 'any')
trainset = trainset.dropna(1, 'any')
results = np.empty([2660,3])

for index, row in trainset.iterrows():
	hg = row['home_team_goal']
	ag = row['away_team_goal']
	results[index] = resultEncoder(hg,ag)

x = tf.placeholder(tf.float32, [None, 21])
W = tf.Variable(tf.zeros([2660,3]))
b = tf.Variable(tf.zeros(3))

y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 3])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()