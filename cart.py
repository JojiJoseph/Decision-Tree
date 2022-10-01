from webbrowser import get
import numpy as np
from collections import Counter

class CART:
    def __init__(self, impurity_function="gini", max_depth = None) -> None:
        self.leaf = False
        self.val = None
        self.left = None
        self.right = None
        self.impurity_function = impurity_function
        self.max_depth = max_depth
        self.depth = 0
    
    def fit(self, X_train, y_train) -> None:
        if len(np.unique(y_train)) == 1:
            self.leaf = True
            self.val = y_train[0]
            return
        else:
            self.leaf = False
        if self.max_depth and self.depth >= self.max_depth:
            self.leaf = True
            if self.impurity_function in ["gini", "entropy"]:
                best_count = 0
                for t in np.unique(y_train):
                    if np.sum(y_train==t) > best_count:
                        best_count = np.sum(y_train==t)
                        self.val = t
            elif self.impurity_function == "mse":
                self.val = np.mean(y_train)
            return
            

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        n_features = X_train.shape[1]
        impurity = self.calc_impurity(y_train)
        best_feature = 0
        best_feature_thresh = None
        best_impurity = 0
        for i in range(n_features):
            feature_vals = np.unique(X_train[:,i]) # unique returns sorted array. So no need to sort again
            thresholds = (feature_vals[:-1] + feature_vals[1:])/2.0

            for thresh in thresholds:
                try:
                    left_y = y_train[X_train[:,i] <= thresh]
                except:
                    print(thresh,X_train[:,i] <= thresh , y_train.shape)
                right_y = y_train[X_train[:,i] > thresh]
                current_impurity = impurity - len(left_y)/len(X_train)*self.calc_impurity(left_y) - len(right_y)/len(X_train) * self.calc_impurity(right_y)
                if  current_impurity >= best_impurity:
                    best_impurity = current_impurity
                    best_feature = i
                    best_feature_thresh = thresh

        self.best_feature = best_feature
        self.best_feature_thresh = best_feature_thresh
        
        self.left = CART(impurity_function=self.impurity_function, max_depth=self.max_depth)
        self.left.depth = self.depth+1
        self.left.fit(X_train[X_train[:,best_feature] <= best_feature_thresh], y_train[X_train[:,best_feature] <= best_feature_thresh])
        
        self.right = CART(impurity_function=self.impurity_function, max_depth=self.max_depth)
        self.right.depth = self.depth+1
        self.right.fit(X_train[X_train[:,best_feature] > best_feature_thresh], y_train[X_train[:,best_feature] > best_feature_thresh])



    def calc_impurity(self, y_train) -> float:
        y_train = np.array(y_train)
        if self.impurity_function == "gini":
            impurity = 1
            classes = np.unique(y_train)
            l = len(y_train)
            for c in classes:
                lc = len(y_train[y_train == c])
                impurity -= (lc/l)**2
            return impurity
        elif self.impurity_function == "mse":
            return np.mean((y_train-np.mean(y_train))**2)
        elif self.impurity_function == "entropy":
            counter = Counter(y_train)
            l = len(y_train)
            impurity = 0
            for k in counter:
                impurity -= counter[k]/l *np.log2(counter[k]/l)
            return impurity

    def predict(self, X_test):
        # return None
        X_test = np.array(X_test)
        y_pred = []
        # print(X_test.shape)
        # print
        for i in range(len(X_test)):
            X = X_test[i]
            # print(X)
            y_pred.append(self.predict_individual(X))
        # print(len(y_pred))
        # print("Finish")
        return np.array(y_pred)

    def predict_individual(self, X):
        # print("$",X)
        if self.leaf:
            return self.val
        if X[self.best_feature] <= self.best_feature_thresh:
            return self.left.predict_individual(X)
        else:
            return self.right.predict_individual(X)


def get_explanation(tree_root):
    # print(tree_root)
    # if not tree_root:
    #     return ""
    # print(tree_root.leaf, tree_root.depth)
    if tree_root.leaf:
        return "|   " * (tree_root.depth) + "|--- value: " + str(tree_root.val) + "\n"
    return "|   " * (tree_root.depth) + "|--- " + f"feature_{tree_root.best_feature}<={tree_root.best_feature_thresh}:\n{get_explanation(tree_root.left)}" \
        + "|   " * (tree_root.depth) + "|--- " + f"feature_{tree_root.best_feature}>{tree_root.best_feature_thresh}:\n{get_explanation(tree_root.right)}"