'''
Author: Jiaqi Li
Class: CS451 Course Project
Data: Global Fishing Watch
'''

#%% load up the data
import csv
import arrow

num_var = ["length_m_gfw","engine_power_kw_gfw", "tonnage_gt_gfw"]
str_var = ["date","flag_gfw","vessel_class_gfw"]

examples = [] # training data collected from 2018
ys = []       # label is accumulative fishing hours for 2019
with open("data/2018_West_Africa.csv") as fp:
    rows = csv.reader(fp)
    header = next(rows)
    for row in rows:
        entry = dict(zip(header, row)) # glue them into a dict
        if entry["fishing_hours_2018"] == "NA" or entry["fishing_hours_2019"] == "NA":
            continue
        ys.append(float(entry["fishing_hours_2019"]))
        #print(entry) # print that dict
        geometry = entry["geometry"].split(', ')
        lat = float(geometry[0][2:])
        lon = float(geometry[1].replace(")", ""))
        temp = {}
        temp["lat"] = lat
        temp["lon"] = lon
        #date = arrow.get(entry["date"], 'MM-DD-YYYY')
        #temp["date"] = date
        for name in str_var:
            if entry[name] == 'NA':
                continue
            else:
                temp[name] = entry[name]
        for name in num_var:
            if entry[name] == 'NA':
                continue
            else:
                temp[name] = float(entry[name])
        #temp["self_reported_fishing_vessel"] = entry["self_reported_fishing_vessel"] == "TRUE"
        examples.append(temp)


#%% vectorize:
from sklearn.feature_extraction import DictVectorizer

feature_numbering = DictVectorizer(sort=True, sparse=False)
feature_numbering.fit(examples)
X = feature_numbering.transform(examples)
print("Features as {} matrix.".format(X.shape))

del examples

#%% Split data
from sklearn.model_selection import train_test_split
import numpy as np

RANDOM_SEED = 12345678

y = np.array(ys)
# split off 10% for train/validate (tv) pieces.
X_tv, rX_test, y_tv, y_test = train_test_split(
    X, y, train_size=0.1, shuffle=True, random_state=RANDOM_SEED
)
# split off 50% train, validate from (tv) pieces.
rX_train, rX_vali, y_train, y_vali = train_test_split(
    X_tv, y_tv, train_size=0.5, shuffle=True, random_state=RANDOM_SEED
)

#%% use MinMaxScaler to scale down X
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaling = StandardScaler()
X_train = scaling.fit_transform(rX_train)
X_vali = scaling.transform(rX_vali)
X_test = scaling.transform(rX_test)

del ys, X, y, y_tv, X_tv, rX_train, rX_vali, rX_test

print(X_train.shape, X_vali.shape, X_test.shape)


#%% Out-of-the-box model performances. Do I need a linear or nonlinear model?
#%% model Experiments
print("\n##### Do I need a linear or nonlinear model? #####")

from sklearn.utils import resample
# sample a much smaller size from the training data test different models
# and their hyperparameters
X_temp, y_temp = resample(
    X_train, y_train, n_samples=1500, replace=False
)

print("training size for models: ", X_temp.shape)

# stdlib:
from dataclasses import dataclass
import json
from typing import Dict, Any, List
from sklearn.base import RegressorMixin

#%% Define & Run Experiments
@dataclass
class ExperimentResult:
    vali_acc: float
    params: Dict[str, Any]
    model: RegressorMixin

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

def consider_decision_trees():
    print("Consider Decision Tree")
    performances: List[ExperimentResult] = []

    for rnd in range(3):
        for crit in ["poisson", "mse", "friedman_mse", "mae"]:
            for d in range(15, 25, 1):
                params = {
                    "criterion": crit,
                    "max_depth": d,
                    "random_state": rnd,
                }
                f = DecisionTreeRegressor(**params)
                f.fit(X_temp, y_temp)
                vali_acc = f.score(X_vali, y_vali)
                result = ExperimentResult(vali_acc, params, f)
                performances.append(result)
    return max(performances, key=lambda result: result.vali_acc)

def consider_knn():
    print("Consider knn")
    performances: List[ExperimentResult] = []

    for weight in ["uniform", "distance"]:
        for n in range(5, 13):
            params = {
                "n_neighbors": n,
                "weights": weight,
            }
            f = KNeighborsRegressor(**params)
            f.fit(X_temp, y_temp)
            vali_acc = f.score(X_vali, y_vali)
            result = ExperimentResult(vali_acc, params, f)
            performances.append(result)
    return max(performances, key=lambda result: result.vali_acc)

def consider_neural_net() -> ExperimentResult:
    print("Consider Multi-Layer Perceptron")
    performances: List[ExperimentResult] = []

    for rnd in range(5):
        for solver in ["lbfgs", "sgd", "adam"]:
                for alpha in range(1, 5):
                    params = {
                        "hidden_layer_sizes": (500,),
                        "random_state": rnd,
                        "solver": solver,
                        "alpha": alpha*0.0001,
                        "max_iter": 5000,
                    }
                    f = MLPRegressor(**params)
                    f.fit(X_temp, y_temp)
                    vali_acc = f.score(X_vali, y_vali)
                    result = ExperimentResult(vali_acc, params, f)
                    print(result)
                    performances.append(result)
    return max(performances, key=lambda result: result.vali_acc)

dtree = consider_decision_trees()
knn = consider_knn()
#mlp = consider_neural_net() ## <- mlp is too slow to run and sgd spits out negative scores

print("\nBest DTree:\n", dtree)
print("Best knn:\n", knn)
#print("Best MLP:\n", mlp)

## result:
#Best DTree:
# ExperimentResult(vali_acc=0.911741239003447, params={'criterion': 'friedman_mse', 'max_depth': 21, 'random_state': 0}, model=DecisionTreeRegressor(criterion='friedman_mse', max_depth=21, random_state=0))
#Best knn:
# ExperimentResult(vali_acc=0.7033403492268693, params={'n_neighbors': 8, 'weights': 'uniform'}, model=KNeighborsRegressor(n_neighbors=8))

# Linear model does not work for this dataset. Features are highly correlated, making the linear model
# very unstable. Large number of iterations is needed to reach the depth of optimization for linear modesl,
# and it is too slow to train.

del X_temp, y_temp

from shared import simple_boxplot, bootstrap_regressor

simple_boxplot(
    {
        "Decision Tree": bootstrap_regressor(dtree.model, X_vali, y_vali),
        "knn": bootstrap_regressor(knn.model, X_vali, y_vali),
        #"MLP/NN": bootstrap_accuracy(mlp.model, X_vali, y_vali),
    },
    title="Validation Accuracy",
    xlabel="Model",
    ylabel="Mean Squared Error",
    save="graphs/project/model-cmp.png",
)

## Decision tree performs better than knn for this dataset. The bootstrapped boxplot shows
# that this dataset has rather high quality without many outliers and much variance.

del dtree, knn



#%% Is my dataset large enough?
#%% Compute performance for each % of training data

print("\n##### Is my dataset large enough? #####")
N = len(y_train)
print(N)
num_trials = 80
percentages = list(range(20, 100, 20)) ## <- quite possibly this amount is enough
percentages.append(100)
scores = {}
acc_mean = []
acc_std = []

for train_percent in percentages:
    n_samples = int((train_percent / 100) * N)
    print("{}% == {} samples...".format(train_percent, n_samples))
    label = "{}".format(n_samples, train_percent)

    # So we consider num_trials=100 subsamples, and train a model on each.
    scores[label] = []
    for i in range(num_trials):
        X_sample, y_sample = resample(
            X_train, y_train, n_samples=n_samples, replace=False
        )  # type:ignore
        clf = DecisionTreeRegressor(max_depth = 19)
        #clf = SGDRegressor(random_state=RANDOM_SEED + n_samples + i)
        clf.fit(X_sample, y_sample)
        scores[label].append(clf.score(X_vali, y_vali))
    acc_mean.append(np.mean(scores[label]))
    acc_std.append(np.std(scores[label]))

# line plot with shaded variance regions
import matplotlib.pyplot as plt

# convert our list of means/std to numpy arrays to add & subtract them.
means = np.array(acc_mean)
std = np.array(acc_std)
# plot line from means
plt.plot(percentages, acc_mean, "o-")
# plot area from means & stddev
plt.fill_between(percentages, means - std, means + std, alpha=0.2)

# Manage axes/show:
plt.xlabel("Percent Training Data")
plt.ylabel("Mean Accuracy")
plt.xlim([0, 100])
plt.title("Shaded Accuracy Plot")
plt.savefig("graphs/project/fishing-area-Accuracy.png")
plt.show()

#%% boxplot to show the learning curve training data
simple_boxplot(
    scores,
    "Learning Curve",
    xlabel="Percent Training Data",
    ylabel="Accuracy",
    save="graphs/project/fishing-boxplots-Accuracy.png",
)

# As the size of the dataset gets bigger, the accuracy of the decision tree model becomes a lot
# higher.


print("\n##### Thoughts on feature design / usefulness #####")
print("see comments")

# I do not have a lot of features so it is suitable to use methods in p10 to figure out the importances
# of each feature with a smaller subset of the training data. Further datapoints to add could include ecological
# variables. I don't have much insights as to what variables are more valuable than others, since I don't know
# fishing patterns.

# (1) should I group data by week to make the time series more reliable?
# The fact that my dataset spans the entire year seems to really assist the accuracy of the model
# (which also intuitively make sense). However, splitting the data by week/month would confirm this intuition.
# (2) what are the opportunities for train/test splitting on this data?
# (3) Should I be using K-fold cross-validation?
# This is definitley possible for my dataset, because of the large size of my data. I trained my models above
# on a very small subset of the dataset. The next step wcould be attempting k-fold cross-validation on the entire
# dataset to see if the hyperparameters derived still hold.
