import numpy as np
import BagLearner as bl
import LinRegLearner as lin
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.learners = np.repeat(bl.BagLearner(lin.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False), 20)
    def author(self):
        return "cchan313"  # replace tb34 with your Georgia Tech username
    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            learner.add_evidence(data_x, data_y)
    def query(self, points):
        for learner in self.learners:
            result = learner.query(points)
if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
