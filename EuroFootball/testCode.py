import pandas as pd
import numpy as np
import tensorflow as tf


#Used to map results
def resultEncoder(hg, ag):
    if hg > ag:
        return np.array([1, 0, 0])
    elif ag > hg:
        return np.array([0, 0, 1])
    else:
        return np.array([0, 1, 0])

#
testset = pd.read_csv('testset.csv')
trainset = pd.read_csv('trainset.csv')
testset = testset.dropna(1, 'any')
trainset = trainset.dropna(1, 'any')
results = np.empty([2660, 3])

# Map the results using resultEncoder
for index, row in trainset.iterrows():
    hg = row['home_team_goal']
    ag = row['away_team_goal']
    results[index] = resultEncoder(hg, ag)

print trainset.apply(lambda x: sum(x.isnull()),axis=0) 