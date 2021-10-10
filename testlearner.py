""""""  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		   	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		   	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		   	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		   	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 		  		  		    	 		 		   		 		  
or edited.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		   	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		   	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		   	 		  		  		    	 		 		   		 		  
"""
import math  		  	   		   	 		  		  		    	 		 		   		 		  
import sys
import numpy as np
import LinRegLearner as lr
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as il
import matplotlib.pyplot as plt

def r2(actual, predict):
    c_matrix = np.corrcoef(actual, predict)
    c = c_matrix[0,1]
    r_sq = c**2
    return r_sq

if __name__ == "__main__":
    sys.stdout = open("p3_results.txt", "w")

    np.seterr(divide="ignore", invalid="ignore")
    if len(sys.argv) != 2:  		  	   		   	 		  		  		    	 		 		   		 		  
        print("Usage: python testlearner.py <filename>")  		  	   		   	 		  		  		    	 		 		   		 		  
        sys.exit(1)  		  	   		   	 		  		  		    	 		 		   		 		  
    inf = open(sys.argv[1])
    if sys.argv[1] == "Data/Istanbul.csv":
        data = np.genfromtxt(sys.argv[1], delimiter=",")
        data = data[1:,1:]
        print(f"Data shape: {data.shape}")
    else:
        data = np.array(
            [list(map(float, s.strip().split(","))) for s in inf.readlines()]
        )
  		  	   		   	 		  		  		    	 		 		   		 		  
    # compute how much of the data is training and testing
    np.random.shuffle(data)
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # separate out training and testing data  		  	   		   	 		  		  		    	 		 		   		 		  
    train_x = data[:train_rows, 0:-1]  		  	   		   	 		  		  		    	 		 		   		 		  
    train_y = data[:train_rows, -1]  		  	   		   	 		  		  		    	 		 		   		 		  
    test_x = data[train_rows:, 0:-1]  		  	   		   	 		  		  		    	 		 		   		 		  
    test_y = data[train_rows:, -1]  		  	   		   	 		  		  		    	 		 		   		 		  

    print()
    print(f"Test X shape: {test_x.shape}")
    print(f"Test Y shape: {test_y.shape}")

    #Experiment 1
    experiment1_in = np.zeros(50)
    experiment1_out = np.zeros(50)

    for i in range(1,51):
    # create a learner and train it
        dt_learner = dt.DTLearner(leaf_size=i, verbose=False)  # create a DTLearner
        dt_learner.add_evidence(train_x, train_y)  # train DTLearner

        if i == 1:
            print()
            print("Experiment 1")
            print(f"Author: {dt_learner.author()}")

        # evaluate in sample (DTLearner)
        dt_pred_y = dt_learner.query(train_x)  # get the predictions
        dt_rmse = math.sqrt(((train_y - dt_pred_y) ** 2).sum() / train_y.shape[0])
        print()
        print("In sample results (DTLearner)")
        print(f"RMSE: {dt_rmse}")
        c = np.corrcoef(dt_pred_y, y=train_y)
        print(f"corr: {c[0, 1]}")
        experiment1_in[i-1] = dt_rmse

        # evaluate out of sample (DTLearner)
        dt_pred_y = dt_learner.query(test_x)  # get the predictions
        dt_rmse = math.sqrt(((test_y - dt_pred_y) ** 2).sum() / test_y.shape[0])
        print()
        print("Out of sample results (DTLearner)")
        print(f"RMSE: {dt_rmse}")
        c = np.corrcoef(dt_pred_y, y=test_y)
        print(f"corr: {c[0, 1]}")
        experiment1_out[i-1] = dt_rmse

    experiment1 = plt.figure(1)
    plt.plot(experiment1_in, label="In-sample RMSE")
    plt.plot(experiment1_out, label="Out-sample RMSE")
    plt.legend(loc="best")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.title("Experiment 1: DTLearner")
    plt.savefig("p3_Experiment1.png")

    # Experiment 2
    experiment2_in = np.zeros(50)
    experiment2_out = np.zeros(50)

    for i in range(1, 51):
        # create a learner and train it
        bag_learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": i}, bags=10, boost=False, verbose=False)
        bag_learner.add_evidence(train_x, train_y)  # train BagLearner
        if i == 1:
            print()
            print("Experiment 2")
            print(f"Author: {bag_learner.author()}")

        # evaluate in sample (DTLearner)
        bag_pred_y = bag_learner.query(train_x)  # get the predictions
        bag_rmse = math.sqrt(((train_y - bag_pred_y) ** 2).sum() / train_y.shape[0])
        print()
        print("In sample results (BagLearner)")
        print(f"RMSE: {bag_rmse}")
        c = np.corrcoef(bag_pred_y, y=train_y)
        print(f"corr: {c[0, 1]}")
        experiment2_in[i - 1] = bag_rmse

        # evaluate out of sample (DTLearner)
        bag_pred_y = bag_learner.query(test_x)  # get the predictions
        bag_rmse = math.sqrt(((test_y - bag_pred_y) ** 2).sum() / test_y.shape[0])
        print()
        print("Out of sample results (BagLearner)")
        print(f"RMSE: {bag_rmse}")
        c = np.corrcoef(bag_pred_y, y=test_y)
        print(f"corr: {c[0, 1]}")
        experiment2_out[i - 1] = bag_rmse

    experiment2 = plt.figure(2)
    plt.plot(experiment2_in, label="In-sample RMSE")
    plt.plot(experiment2_out, label="Out-sample RMSE")
    plt.legend(loc="best")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.title("Experiment 2: BagLearner w/ 10 Bags")
    plt.savefig("p3_Experiment2.png")

    # Experiment 3
    # Metric 1- Mean Absolute Error (MAE)
    experiment3_m1dt_in = np.zeros(50)
    experiment3_m1dt_out = np.zeros(50)
    experiment3_m1rt_in = np.zeros(50)
    experiment3_m1rt_out = np.zeros(50)

    # Metric 2- Coefficient of Determination (R-Squared)
    experiment3_m2dt_in = np.zeros(50)
    experiment3_m2dt_out = np.zeros(50)
    experiment3_m2rt_in = np.zeros(50)
    experiment3_m2rt_out = np.zeros(50)

    for i in range(1, 51):
        # create a learner and train it
        dt_learner = dt.DTLearner(leaf_size=i, verbose=False)  # create a DTLearner
        dt_learner.add_evidence(train_x, train_y)  # train DTLearner
        rt_learner = rt.RTLearner(leaf_size=i, verbose=False)  # create a RTLearner
        rt_learner.add_evidence(train_x, train_y)  # train RTLearner

        if i == 1:
            print()
            print("Experiment 3")
            print(f"Author: {rt_learner.author()}")

        # evaluate in sample (DTLearner)
        dt_pred_y = dt_learner.query(train_x)  # get the predictions
        dt_mae = np.mean(np.abs(train_y - dt_pred_y)) # Mean Absolute Error (MAE)
        dt_r2 = r2(train_y, dt_pred_y)
        print()
        print("In sample results (DTLearner)")
        print(f"MAE: {dt_mae}")
        print(f"R2: {dt_r2}")
        c = np.corrcoef(dt_pred_y, y=train_y)
        print(f"corr: {c[0, 1]}")
        experiment3_m1dt_in[i - 1] = dt_mae
        experiment3_m2dt_in[i - 1] = dt_r2

        # evaluate in sample (RTLearner)
        rt_pred_y = rt_learner.query(train_x)  # get the predictions
        rt_mae = np.mean(np.abs(train_y - rt_pred_y)) # Mean Absolute Error (MAE)
        rt_r2 = r2(train_y, rt_pred_y)
        print()
        print("In sample results (RTLearner)")
        print(f"MAE: {rt_mae}")
        print(f"R2: {rt_r2}")
        c = np.corrcoef(rt_pred_y, y=train_y)
        print(f"corr: {c[0, 1]}")
        experiment3_m1rt_in[i - 1] = rt_mae
        experiment3_m2rt_in[i - 1] = rt_r2

        # evaluate out of sample (DTLearner)
        dt_pred_y = dt_learner.query(test_x)  # get the predictions
        dt_mae = np.mean(np.abs(test_y - dt_pred_y)) # Mean Absolute Error (MAE)
        dt_r2 = r2(test_y, dt_pred_y)
        print()
        print("Out of sample results (DTLearner)")
        print(f"MAE: {dt_mae}")
        print(f"R2: {dt_r2}")
        c = np.corrcoef(dt_pred_y, y=test_y)
        print(f"corr: {c[0, 1]}")
        experiment3_m1dt_out[i - 1] = dt_mae
        experiment3_m2dt_out[i - 1] = dt_r2

        # evaluate out of sample (RTLearner)
        rt_pred_y = rt_learner.query(test_x)  # get the predictions
        rt_mae = np.mean(np.abs(test_y - rt_pred_y)) # Mean Absolute Error (MAE)
        rt_r2 = r2(test_y, rt_pred_y)
        print()
        print("Out of sample results (RTLearner)")
        print(f"MAE: {rt_mae}")
        print(f"R2: {rt_r2}")
        c = np.corrcoef(rt_pred_y, y=test_y)
        print(f"corr: {c[0, 1]}")
        experiment3_m1rt_out[i - 1] = rt_mae
        experiment3_m2rt_out[i - 1] = rt_r2

    experiment3_m1 = plt.figure(3)
    plt.plot(experiment3_m1dt_in, label="DTLearner In-sample MAE")
    plt.plot(experiment3_m1dt_out, label="DTLearner Out-sample MAE")
    plt.plot(experiment3_m1rt_in, label="RTLearner In-sample MAE")
    plt.plot(experiment3_m1rt_out, label="RTLearner Out-sample MAE")
    plt.legend(loc="best")
    plt.xlabel("Leaf Size")
    plt.ylabel("MAE")
    plt.title("Experiment 3: DTLearner vs RTLearner (MAE)")
    plt.savefig("p3_Experiment3_m1.png")

    experiment3_m2 = plt.figure(4)
    plt.plot(experiment3_m2dt_in, label="DTLearner In-sample R-Squared")
    plt.plot(experiment3_m2dt_out, label="DTLearner Out-sample R-Squared")
    plt.plot(experiment3_m2rt_in, label="RTLearner In-sample R-Squared")
    plt.plot(experiment3_m2rt_out, label="RTLearner Out-sample R-Squared")
    plt.legend(loc="best")
    plt.xlabel("Leaf Size")
    plt.ylabel("R-Squared")
    plt.title("Experiment 3: DTLearner vs RTLearner (R-Squared)")
    plt.savefig("p3_Experiment3_m2.png")

    sys.stdout.close()







