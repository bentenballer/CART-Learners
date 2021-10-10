import numpy as np
from numpy import random
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as ins

def test():
    arr = np.genfromtxt("Data/Istanbul.csv", delimiter=",")
    arr = arr[1:,1:]
    print(arr)
    print(arr.shape)
    data_x = arr[:,0:-1]
    data_y = arr[:,-1]
    #learner = dt.DTLearner(leaf_size=1, verbose=False)
    #learner = ins.InsaneLearner(verbose=False)
    learner = bl.BagLearner(learner=dt.DTLearner,kwargs={"leaf_size":1},bags=5,boost=False,verbose=False)
    learner.add_evidence(data_x, data_y)
    test_x = np.array([[1,2],[2,3], [30,10]])
    results = learner.query(test_x)
    #for ml in learner.learners:
    #    print(ml.tree)
    #print(learner.tree)
    #print(results)
    return learner

if __name__ == "__main__":
    test()
