'''
Author: Jiaqi Li
Class: CS451 Course Project
Data: Global Fishing Watch
'''

#%% load up the data
import csv
import arrow

num_var = ["length_m_gfw","engine_power_kw_gfw", "tonnage_gt_gfw"]
str_var = ["flag_gfw","vessel_class_gfw"]

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
print(feature_numbering.feature_names_)
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

# Sample 1500 data points for feature engineering
from sklearn.utils import resample
# sample a much smaller size from the training data test different models
# and their hyperparameters
X_temp, y_temp = resample(
    X_train, y_train, n_samples=1500, replace=False
)
print("training size for models: ", X_temp.shape)



#%% Feature Performance Analysis

from sklearn.ensemble import RandomForestRegressor

# Direct feature-importances (can think of them as how many times a feature was used):
rf = RandomForestRegressor(random_state=123456, n_estimators=100)
rf.fit(X_temp, y_temp)

# loop over each tree and ask them how important each feature was!
importances = dict((name, []) for name in feature_numbering.feature_names_)
for tree in rf.estimators_:
    for name, weight in zip(feature_numbering.feature_names_, tree.feature_importances_):
        importances[name].append(weight)

im = {}
import statistics as st
for name in importances.keys():
    if st.mean(importances[name]) > 0.04:
        im[name] = importances[name]



from shared import simple_boxplot, bootstrap_r2

simple_boxplot(
    im,
    title="Tree Importances",
    ylabel="Decision Tree Criterion Importances",
    save='graphs/project/feature-importance'
)

import typing as T
from dataclasses import dataclass
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

graphs: T.Dict[str, T.List[float]] = {}

@dataclass
class Model:
    vali_score: float
    m: T.Any

def train_and_eval(name, x, y, vx, vy):
    """ Train and Eval a single model. """
    options: T.List[Model] = []
    for d in range(10, 21):
        m = DecisionTreeRegressor(criterion="friedman_mse", max_depth=d, random_state=0)
        m.fit(x, y)
        options.append(Model(m.score(vx, vy), m))

    for n in range(5, 13):
        f = KNeighborsRegressor(n_neighbors=n, weights='uniform')
        f.fit(X_temp, y_temp)
        options.append(Model(m.score(vx, vy), m))

    # pick the best model:
    best = max(options, key=lambda m: m.vali_score)
    # bootstrap its output:
    graphs[name] = bootstrap_r2(best.m, vx, vy)
    # record our progress:
    print("{:20}\t{:.3}\t{}".format(name, np.mean(graphs[name]), best.m))


train_and_eval("Full Model", X_temp, y_temp, X_vali, y_vali)

for fid, fname in enumerate(feature_numbering.feature_names_):
    # one-by-one, delete your features:
    without_X = X_temp.copy()
    without_X[:, fid] = 0.0
    # score a model without the feature to see if it __really__ helps or not:
    train_and_eval("without {}".format(fname), without_X, y_temp, X_vali, y_vali)

# Inline boxplot code here so we can sort by value:
box_names = []
box_dists = []
for (k, v) in sorted(graphs.items(), key=lambda tup: np.mean(tup[1])):
    box_names.append(k)
    box_dists.append(v)

import matplotlib.pyplot as plt

plt.boxplot(box_dists)
plt.xticks(
    rotation=30,
    horizontalalignment="right",
    ticks=range(1, len(box_names) + 1),
    labels=box_names,
)
plt.title("Feature Removal Analysis")
plt.xlabel("Included?")
plt.ylabel("AUC")
plt.tight_layout()
plt.savefig("graphs/project/feature-engineering.png")
plt.show()



### I've already tested multiple models in the last project checkpoint


### As you predicted, I'm getting bored with this project. I'm looking for a
# coding challenge. I will choose a model and implement it in C.
