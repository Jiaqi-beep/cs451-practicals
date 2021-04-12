import csv
import arrow
from tqdm import tqdm
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler


#%% load up the data
examples = []
ys = []

num_var = ["length_m_gfw", "engine_power_kw_gfw", "tonnage_gt_gfw"]
str_var = ["date", "flag_gfw", "vessel_class_gfw"]

with open("data/2019_West_Africa.csv") as fp:
    rows = csv.reader(fp)
    header = next(rows)
    for row in rows:
        # print(header) # print the labels
        # print(row) # print the current row
        entry = dict(zip(header, row))  # glue them into a dict
        if entry["fishing_hours_2019"] == "NA":
            continue
        ys.append(float(entry["fishing_hours_2019"]))
        # print(entry) # print that dict
        geometry = entry["geometry"].split(", ")
        lat = float(geometry[0][2:])
        lon = float(geometry[1].replace(")", ""))
        temp = {}
        temp["lat"] = lat
        temp["lon"] = lon
        # date = arrow.get(entry["date"], "m--YYYY")
        # temp["year"] = date.year
        # temp["month"] = date.month
        for name in str_var:
            if entry[name] == "NA":
                continue
            else:
                temp[name] = entry[name]
        for name in num_var:
            if entry[name] == "NA":
                continue
            else:
                temp[name] = float(entry[name])
        # temp["self_reported_fishing_vessel"] = entry["self_reported_fishing_vessel"] == "TRUE"
        examples.append(temp)

print(examples[3487])

from sklearn.feature_extraction import DictVectorizer

feature_numbering = DictVectorizer(sort=True, sparse=False)
feature_numbering.fit(examples)
X = feature_numbering.transform(examples)

# print(type(X))
print("Features as {} matrix.".format(X.shape))
# print(X)

from sklearn.model_selection import train_test_split

RANDOM_SEED = 12345678

# Numpy-arrays are more useful than python's lists.
y = np.array(ys)
# split off train/validate (tv) pieces.
X_tv, rX_test, y_tv, y_test = train_test_split(
    X, y, train_size=0.5, shuffle=True, random_state=RANDOM_SEED
)
# split off train, validate from (tv) pieces.
rX_train, rX_vali, y_train, y_vali = train_test_split(
    X_tv, y_tv, train_size=0.5, shuffle=True, random_state=RANDOM_SEED
)

scaling = StandardScaler()
X_train = scaling.fit_transform(rX_train)
X_vali = scaling.transform(rX_vali)
X_test = scaling.transform(rX_test)

print(X_train.shape, X_vali.shape, X_test.shape)

params = {"max_depth": 9}


def d_trees():
    print("Consider Decision Tree.")

    f = DecisionTreeRegressor(**params)
    f.fit(X_train, y_train)
    print(f.score(X_train, y_train))
    print(f.score(X_vali, y_vali))


d_trees()


sgd = SGDRegressor()
print("Train SGDRegressor (sgd)")
for iter in tqdm(range(1000)):
    sgd.partial_fit(X_train, y_train)


print("sgdc. Train-Accuracy: {:.3}".format(sgd.score(X_train, y_train)))
print("sgdc. Vali-Accuracy: {:.3}".format(sgd.score(X_vali, y_vali)))
