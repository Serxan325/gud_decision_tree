import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Tree:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.root = None

    def gini(self, target):
        if len(target) == 0:
            return 0

        probabilities = target.value_counts(normalize=True)

        return 1 - (probabilities ** 2).sum()

    def best_split(self, features, target, min_samples_leaf):
        best_gini = self.gini(target)
        best_split = None

        for feature in features:
            if pd.api.types.is_bool_dtype(features[feature]):
                thresholds = [None]

            elif pd.api.types.is_numeric_dtype(features[feature]):
                values = np.sort(features[feature].dropna().unique())
                thresholds = [
                    (values[i] + values[i + 1]) / 2
                    for i in range(len(values) - 1)
                ]

            else:
                continue

            for threshold in thresholds:
                if threshold is None:
                    left_mask = features[feature] == True
                else:
                    left_mask = features[feature] < threshold

                right_mask = ~left_mask

                left_target = target.loc[left_mask]
                right_target = target.loc[right_mask]

                if (len(left_target) < min_samples_leaf or
                        len(right_target) < min_samples_leaf):
                    continue

                weighted_gini = (
                    len(left_target) / len(target) * self.gini(left_target)
                    + len(right_target) / len(target) * self.gini(right_target)
                )

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_split = {
                        "feature": feature,
                        "threshold": threshold,
                        "left_mask": left_mask
                    }

        return best_split

    def grow(self, features, target, depth, max_depth, min_samples_leaf):
        true_count = (target == True).sum()
        false_count = (target == False).sum()

        prediction = true_count > false_count

        leaf = {
            "prediction": prediction
        }

        if (
            true_count == 0
            or false_count == 0
            or depth >= max_depth
            or len(target) < 2 * min_samples_leaf
        ):
            return leaf

        split = self.best_split(features, target, min_samples_leaf)

        if split is None:
            return leaf

        feature = split["feature"]
        threshold = split["threshold"]
        left_mask = split["left_mask"]

        left_features = features.loc[left_mask]
        right_features = features.loc[~left_mask]

        left_target = target.loc[left_mask]
        right_target = target.loc[~left_mask]

        if threshold is None:
            condition = True
        else:
            condition = threshold

        node = {
            "feature": feature,
            "threshold": condition,
            "left": self.grow(
                left_features,
                left_target,
                depth + 1,
                max_depth,
                min_samples_leaf
            ),
            "right": self.grow(
                right_features,
                right_target,
                depth + 1,
                max_depth,
                min_samples_leaf
            )
        }

        return node

    def __call__(self, limit=3, max_depth=10):
        self.root = self.grow(
            self.X_train,
            self.y_train,
            depth=0,
            max_depth=max_depth,
            min_samples_leaf=limit
        )

        return self.root

    def sampler(self, root, sample):
        node = root

        while "prediction" not in node:
            feature = node["feature"]
            threshold = node["threshold"]

            if threshold is True:
                condition = sample[feature] == True
            else:
                condition = sample[feature] < threshold

            if condition:
                node = node["left"]
            else:
                node = node["right"]

        return node["prediction"]

    def predict(self, X):
        return pd.Series(
            [self.sampler(self.root, sample) for _, sample in X.iterrows()],
            index=X.index
        )

data = pd.read_csv("example.csv")

X = data.drop(columns=["target"])
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tree = Tree(X_train, y_train)

tree(limit=4)

predictions = tree.predict(X_test)

accuracy = (predictions == y_test.to_numpy()).mean()

predictions_indexed = predictions.loc[y_test.index]

y_test_indexed = y_test.reset_index(drop=True)
predictions_indexed = predictions_indexed.reset_index(drop=True)

i = 2

print(predictions_indexed[i])
print(y_test_indexed[i])

#print("Predictions:", predictions)
#print("Actual:", y_test.to_numpy())
print("Accuracy:", accuracy)
