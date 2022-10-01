import numpy as np

class CART:
    def __init__(self) -> None:
        self.leaf = True
        self.val = -1
        self.left = None
        self.right = None
    def fit(self, X_train, y_train):
        # print(len(X_train))
        # y_train = y_train.reshape((-1,1))
        # self.leaf = False
        if len(np.unique(y_train)) == 1:
            self.leaf = True
            self.val = y_train[0]
            return
        else:
            self.leaf = False

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
            # if len(thresholds) == 0:
            #     print("feature_vals", feature_vals, "thresholds", thresholds)
            #     exit()
            # print(thresholds)
            # if not thresholds:

            for thresh in thresholds:
                try:
                    left_y = y_train[X_train[:,i] <= thresh]
                except:
                    print(thresh,X_train[:,i] <= thresh , y_train.shape)
                right_y = y_train[X_train[:,i] > thresh]
                current_impurity = impurity - len(left_y)/len(X_train)*self.calc_impurity(left_y) + len(right_y)/len(X_train) * self.calc_impurity(right_y)
                if  current_impurity >= best_impurity:
                    best_impurity = current_impurity
                    best_feature = i
                    best_feature_thresh = thresh
        if not best_feature_thresh:
            self.leaf = True
            best_count = 0
            for t in np.unique(y_train):
                if np.sum(y_train==t) > best_count:
                    best_count = np.sum(y_train==t)
                    self.val = t

            return

        self.best_feature = best_feature
        self.best_feature_thresh = best_feature_thresh
        self.left = CART()
        self.left.fit(X_train[X_train[:,best_feature] <= best_feature_thresh], y_train[X_train[:,best_feature] <= best_feature_thresh])
        self.right = CART()
        self.right.fit(X_train[X_train[:,best_feature] > best_feature_thresh], y_train[X_train[:,best_feature] > best_feature_thresh])



    def calc_impurity(self, y_train):
        y_train = np.array(y_train)
        impurity = 1
        classes = np.unique(y_train)
        l = len(y_train)
        for c in classes:
            lc = len(y_train[y_train == c])
            impurity -= (lc/l)**2
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