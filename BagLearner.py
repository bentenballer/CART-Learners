import numpy as np
from numpy import random

class BagLearner(object):
    """
    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """
    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
        """
        Constructor method
        """
        self.learners = []
        for i in range(0,bags):
            self.learners.append(learner(**kwargs))
        self.bags = bags


    def bootstrap(self, data):
        bag = np.zeros(data.shape[1])
        for i in range(data.shape[0]):
            i = random.randint(0, data.shape[0])
            bag = np.vstack((bag, data[i]))
        bag = bag[1:]
        return bag

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner
        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        data = np.concatenate((data_x, data_y[:,None]), axis=1)
        for learner in self.learners:
            bag = self.bootstrap(data)
            bag_x = bag[:,:-1]
            bag_y = bag[:,-1]
            learner.add_evidence(bag_x, bag_y)

    def query(self, points):
        """
        Estimate a set of test points given the model we built.
        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        results_avg = np.zeros(points.shape[0])
        stack = np.zeros(points.shape[0])
        for learner in self.learners:
            result = learner.query(points)
            stack = np.vstack((stack, result))
        stack = stack[1:]
        for i in range(results_avg.shape[0]):
            results_avg[i] = np.mean(stack[:,i])
        return results_avg
