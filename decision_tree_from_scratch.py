try:
    from sklearn import datasets
    import pandas as pd
    import numpy as np
    import math
    import warnings
    warnings.filterwarnings("ignore")

    iris = datasets.load_iris()

    arr = np.array(iris)
    df = pd.DataFrame(iris.data)
    y = pd.DataFrame(iris.target)
    df.columns = ["sl", "sw", 'pl', 'pw']


    # Function to find label for a value
    # if MIN_Value <=val < (m + Mean_Value) / 2 then it is assigned label a
    # if (m + Mean_Value) <=val < Mean_Value then it is assigned label b
    # if (Mean_Value) <=val < (Mean_Value + MAX_Value)/2 then it is assigned label c
    # if (Mean_Value + MAX_Value)/2 <=val <= MAX_Value  then it is assigned label d

    def label(val, *boundaries):
        if (val < boundaries[0]):
            return 'a'
        elif (val < boundaries[1]):
            return 'b'
        elif (val < boundaries[2]):
            return 'c'
        else:
            return 'd'


    # Function to convert a continuous data into labelled data
    # There are 4 lables  - a, b, c, d

    def toLabel(df, old_feature_name):
        second = df[old_feature_name].mean()
        minimum = df[old_feature_name].min()
        first = (minimum + second) / 2
        maximum = df[old_feature_name].max()
        third = (maximum + second) / 2
        return df[old_feature_name].apply(label, args=(first, second, third))


    # Convert all columns to labelled data
    df['sl_labeled'] = toLabel(df, 'sl')
    df['sw_labeled'] = toLabel(df, 'sw')
    df['pl_labeled'] = toLabel(df, 'pl')
    df['pw_labeled'] = toLabel(df, 'pw')
    df.drop(['sl', 'sw', 'pl', 'pw'], axis=1, inplace=True)
    unused_features = list(df.columns)
    df["flower_type"] = y
    class Node:
        def __init__(self, features, df, split_feature, level, entropy):
            # split_feature stores the feature, on which the respective node will be split
            # features; contains the unused features for the respective node
            self.one = None
            self.two = None
            self.three = None
            self.four = None
            self.features = features
            self.df = df
            self.split_feature = split_feature
            self.level = level
            self.entropy = entropy
            self.info()
            print("variety of flowers is", len(set(df["flower_type"])))
            if len(self.features) == 0:
                print("Reached leaf Node")
                print()
                print()
            elif len(set(df["flower_type"])) == 1:
                print("Reached leaf Node")
                print()
                print()
            else:
                self.split()

        def info(self):
            print("level", self.level)
            classes = set(self.df["flower_type"])
            for c in classes:         #calculating entropy for that node
                print("count of ", c, " = ", len(self.df[self.df["flower_type"] == c]))
                probability = len(self.df[self.df["flower_type"] == c]) / len(self.df["flower_type"])
                self.entropy -= probability * math.log(probability, 2)
            print("current entropy is = ", self.entropy)

        def split(self):
            best_feature = ""
            max_gain_ratio = -99999999
            for f in self.features:  # left out features upon which the node can be split
                info_req = 0
                info_gain = 0
                split_info = 0
                gain_ratio = 0
                labels = set(self.df[f])
                for l in labels:                    # labels are a,b,c,d
                    prob1 = len(self.df[self.df[f] == l]) / len(self.df["flower_type"])
                    split_info -= prob1 * math.log(prob1)       # calculating split info
                    classes = set(self.df[self.df[f] == l]["flower_type"])
                    entropy = 0
                    for c in classes:              # classes are the labels(0,1,2) of flower type
                        prob2 = len(self.df[self.df[f] == l][self.df["flower_type"] == c]["flower_type"]) / len(
                            self.df[self.df[f] == l]["flower_type"])
                        if len(self.df[self.df[f] == l]["flower_type"])==0:
                            print("no records left")
                        entropy -= prob2 * math.log(prob2, 2)       # calculating entropy
                    info_req += prob1 * entropy
                info_gain = self.entropy - info_req
                gain_ratio = info_gain / split_info               #  calculating gain_ratio, i.e. info_gain/split/info
                if (gain_ratio > max_gain_ratio):
                    max_gain_ratio = gain_ratio
                    best_feature = f
            print("splitting on feature ", best_feature, " with gain ratio ", gain_ratio)
            print()
            print()
            self.split_feature = best_feature
            features = self.features
            features.remove(best_feature)
            if len(self.df[self.df[self.split_feature] == "d"])!=0:       # if no such combination exists in the dataframe, then child not will be created
                self.one = Node(features, self.df[self.df[self.split_feature] == "d"], None, self.level + 1, 0)
            if len(self.df[self.df[self.split_feature] == "c"])!=0:
                self.one = Node(features, self.df[self.df[self.split_feature] == "c"], None, self.level + 1, 0)
            if len(self.df[self.df[self.split_feature] == "b"])!=0:
                self.one = Node(features, self.df[self.df[self.split_feature] == "b"], None, self.level + 1, 0)
            if len(self.df[self.df[self.split_feature] == "a"])!=0:
                self.one = Node(features, self.df[self.df[self.split_feature] == "a"], None, self.level + 1, 0)
    Root = Node(unused_features, df, "", 0, 0)
except Exception as e:
    print("exception is",e)
