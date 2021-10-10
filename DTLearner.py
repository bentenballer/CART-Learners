import numpy as np

class DTLearner(object):
    """
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    def __init__(self, leaf_size=1, verbose=False):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		   	 		  		  		    	 		 		   		 		  
        """
        self.leaf_size = leaf_size


    def buildtree(self, data):
        if data.shape[0] == 1: return np.array([-1, data[0,-1], -1 , -1])
        elif np.all(data[:,-1] == data[:,-1][0]): return np.array([-1, data[0,-1], -1, -1])
        elif data.shape[0] <= self.leaf_size: return np.array([-1, np.median([data[:,-1]]), -1, -1])
        else:
            corr = np.zeros(data.shape[1]-1)
            j = 0
            for idx in range(0,data.shape[1]-1):
                x = data[:,idx]
                y = data[:,-1]
                correlation = np.corrcoef(x,y)
                corr[j] = correlation[0][1]
                j = j + 1
            i = np.argmax(corr)
            split_val = np.median(data[:,i])
            if np.all(data[:,i] <= split_val): return np.array([-1, np.median([data[:,-1]]), -1, -1])
            left_tree = self.buildtree(data[data[:,i] <= split_val])
            right_tree = self.buildtree(data[data[:,i] > split_val])
            if left_tree.ndim==1:root = np.array([i, split_val, 1, 2])
            else:root = np.array([i, split_val, 1, left_tree.shape[0] + 1])
            append = (root, left_tree, right_tree)
            tree = np.vstack(append)
            return tree

    def add_evidence(self, data_x, data_y):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Add training data to learner  		  	   		   	 		  		  		    	 		 		   		 		     		   	 		  		  		    	 		 		   		 		  
        :param data_x: A set of feature values used to train the learner  		  	   		   	 		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		   	 		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        """
        # decision tree algorithm (Jr Quinlan)
        data = np.concatenate((data_x, data_y[:,None]),axis=1)
        tree = self.buildtree(data)
        self.tree = tree

    def query(self, points):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		   	 		  		  		    	 		 		   		 		    	   		   	 		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		   	 		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        """
        dimension = np.ndim(points)
        if dimension==1: results = np.zeros(1)
        else: results = np.zeros(points.shape[0])
        if dimension==1:
            traverse = True
            i = 0
            while traverse:
                node = self.tree[i]
                if node[0] == -1:
                    results[0] = node[1]
                    traverse = False
                else:
                    factor = points[int(node[0])]
                    splitval = node[1]
                    left = node[2]
                    right = node[3]
                    if factor <= splitval:i = i + int(left)
                    else: i = i + int(right)
        else:
            j = 0
            for point in points:
                traverse = True
                i = 0
                while traverse:
                    if np.ndim(self.tree) == 1: node = self.tree
                    else: node = self.tree[i]
                    if node[0] == -1:
                        results[j] = node[1]
                        j = j + 1
                        traverse = False
                    else:
                        factor = point[int(node[0])]
                        splitval = node[1]
                        left = node[2]
                        right = node[3]
                        if factor <= splitval:i = i + int(left)
                        else: i = i + int(right)
        return results
