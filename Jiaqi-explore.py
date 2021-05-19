'''
Jiaqi Li

Implement decision tree

'''

from dataclasses import dataclass
from typing import List
from abc import ABC, abstractmethod

import pandas as pd

#%%
# Create dummy training data

d = {'x': [66, 106, 51, 150, 288, 319, 368, 437], 
		'y': [140, 166, 241, 263, 169, 130, 213, 141],
		'y_actual': [True, True, True, True, False, False, False, False]}
data = pd.DataFrame(data=d)

print(data)



#%% Decision Tree class
class DTreeNode(ABC):
    """ DTreeNode is an abstract class. All nodes in the Decision Tree can 'predict'. """

    @abstractmethod
    def predict(self, x: List[float]) -> float:
        raise ValueError("This is an 'abstract' class!")


@dataclass
class DTreeBranch(DTreeNode):
    """ We've decided to split based on 'feature' at the value 'split_at'. """

    feature: int
    split_at: float
    less_than: DTreeNode
    greater_than: DTreeNode

    def predict(self, x: pd.DataFrame) -> float:
        if x[self.feature] < self.split_at:
            return self.less_than.predict(x)
        else:
            return self.greater_than.predict(x)


@dataclass
class DTreeLeaf(DTreeNode):
    """ We've decided to stop splitting; here's our estimate of the answer. """

    estimate: float

    def predict(self, x: List[float]) -> float:
        return self.estimate



def gini_impurity(data: pd.DataFrame) -> float:
    """
    The standard version of gini impurity sums over the classes
    """

    p_true = sum(data["y_actual"]) / len(data)
    p_false = 1.0 - p_true
    return p_true * (1 - p_true) + p_false * (1 - p_false)


def impurity_of_split(data: pd.DataFrame, feature: str, split: float) -> float:
    """ Find gini index with given split """

    # if we split on this feature at split, we get these two leaves:
    j = k = 0
    smaller = pd.DataFrame(columns = ["x", "y", "y_actual"])
    bigger = pd.DataFrame(columns = ["x", "y", "y_actual"])

    for i in range(len(data)):
    	#print(i)
    	if data.iloc[i][feature] < split:
    		smaller.loc[j] = data.iloc[i]
    		j += 1
    	else:
    		bigger.loc[k] = data.iloc[i]
    		k += 1

    # weight impurity of left and right by size of dataset; this makes tiny splits less awesome.
    wSmaller = len(smaller) / len(data)
    wBigger = len(bigger) / len(data)

    return wSmaller*gini_impurity(smaller) + wBigger*gini_impurity(bigger)


def find_candidate_splits(data: pd.DataFrame, feature: str) -> List[float]:
	midpoints = []
	for i in range(len(data) - 1):
		first = data.iloc[i][feature]
		second = data.iloc[i+1][feature]
		mid = (first + second) * 0.5
		midpoints.append(mid)
	return midpoints


def make_leaf_node(data: pd.DataFrame):
       countTrue = sum(data['y_actual'])
       countFalse = len(data) - countTrue
       prediction = False
       if countTrue > countFalse:
            prediction = True
       return DTreeLeaf(prediction)


def train(data: pd.DataFrame):

	countTrue = sum(data['y_actual'])
	countFalse = len(data) - countTrue
	if len(data) == 0 or countTrue == 0 or countFalse == 0:
		return make_leaf_node(data)

	best_score = 10
	best_feature = ""
	best_split = 0.0

	for feature in ["x", "y"]:
		splits = find_candidate_splits(data, feature)
		for split in splits:
			score = impurity_of_split(data, feature, split)
			if best_score == 10 or score <= best_score:
				best_score = score
				best_feature = feature
				best_split = split

	j = k = 0
	left = pd.DataFrame(columns = ["x", "y", "y_actual"])
	right = pd.DataFrame(columns = ["x", "y", "y_actual"])

	#print("score is {}, data size is {}, best_split is {}, best_feature is {}".format(score, len(data), best_split, best_feature))
	for i in range(len(data)):
		if data.iloc[i][best_feature] < best_split:
			left.loc[j] = data.iloc[i]
			j += 1
		else:
			right.loc[k] = data.iloc[i]
			k += 1

	if len(left) == 0 or len(right) == 0:
		return make_leaf_node(data)

	return DTreeBranch(best_feature, best_split, train(left), train(right))



#m = DTreeBranch(0, 0.5, DTreeLeaf(1.0), DTreeLeaf(0.0))

m = train(data)


assert True == m.predict({"x": 1, "y": 10})
assert True == m.predict({"x": 123, "y": 200})
assert False == m.predict({"x": 500, "y": 200})
assert False == m.predict({"x": 432, "y": 200})

print("no assertion errors")