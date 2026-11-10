# Decision Tree Classifier from Scratch

A simple implementation of a **binary classification Decision Tree** built from scratch using Python, NumPy, and Pandas.

The project demonstrates how a Decision Tree learns from data by finding the best feature splits using **Gini Impurity**, recursively building a tree, and making predictions on unseen data.

## Features

- **Gini Impurity:** Measures the impurity of a dataset.
- **Best Split Selection:** Finds the optimal feature and threshold that minimize weighted Gini impurity.
- **Recursive Tree Construction:** Builds the decision tree by repeatedly splitting the dataset.
- **Boolean Features:** Supports boolean features using `True` and `False` splits.
- **Numerical Features:** Supports numerical features by testing thresholds between sorted unique values.
- **Stopping Conditions:** Prevents unnecessary splitting using maximum depth, minimum samples per leaf, and pure nodes.
- **Prediction:** Uses the constructed tree to classify new samples.

## How It Works

The algorithm follows these steps:

1. Calculate the Gini impurity of the current dataset.
2. Iterate through the available features.
3. For numerical features, test possible thresholds between unique values.
4. For boolean features, split the data into `True` and `False` groups.
5. Calculate the weighted Gini impurity for each possible split.
6. Select the split that produces the lowest impurity.
7. Repeat the process recursively until a stopping condition is reached.
8. Assign a prediction to each leaf based on the majority class.

### Gini Impurity

Gini Impurity measures how mixed the classes are in a dataset.

The formula is:

$$
Gini = 1 - \sum_{i=1}^{n} p_i^2
$$

Where:
- $p_i$ is the proportion of samples belonging to class $i$.
- A Gini value of 0 means the dataset is completely pure.

For each possible split, the algorithm calculates:

$$
Gini_{split} =
\frac{N_L}{N} \times Gini_L +
\frac{N_R}{N} \times Gini_R
$$

Where:
- $N$ is the total number of samples.
- $N_L$ is the number of samples in the left child.
- $N_R$ is the number of samples in the right child.
- $Gini_L$ and $Gini_R$ are the Gini impurities of the two child nodes.

The split with the lowest weighted Gini impurity is selected.

## Project Structure

```text
decision-tree/
│
├── main.py
├── example.csv
└── README.md
```

- `main.py` — Decision Tree implementation and training example.
- `example.csv` — Dataset used for training and testing.
- `README.md` — Project documentation.

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn

Install the required libraries:

```bash
pip install numpy pandas scikit-learn
```

## Usage

### 1. Prepare your dataset

Create a CSV file containing your features and a binary target column named `target`.

Example:

| Feature 1 | Feature 2 | Feature 3 | target |
|---|---|---|---|
| 25 | True | 120 | True |
| 32 | False | 90 | False |
| 45 | True | 150 | True |
| 21 | False | 80 | False |

The target must contain boolean values (`True` and `False`).

### 2. Load your dataset

```python
data = pd.read_csv("example.csv")

X = data.drop(columns=["target"])
y = data["target"]
```

### 3. Split the dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 4. Train the Decision Tree

```python
tree = Tree(X_train, y_train)

tree(limit=4, max_depth=10)
```

Parameters:
- `limit` — Minimum number of samples required in each leaf.
- `max_depth` — Maximum depth of the tree.

### 5. Make predictions

```python
predictions = tree.predict(X_test)
```

### 6. Evaluate the model

```python
accuracy = (predictions.to_numpy() == y_test.to_numpy()).mean()

print("Accuracy:", accuracy)
```

## Limitations

This is an educational implementation, so it has some limitations:

- Supports binary classification only.
- Target values must be boolean (`True` and `False`).
- Supports boolean and numerical features.
- Categorical and text features are not supported directly.
- Missing values and NaN handling are not implemented.
- Does not include pruning or other advanced tree optimization techniques.
- Uses a simple majority-class prediction at each leaf.

## Purpose

The main goal of this project is to understand the internal mechanics of Decision Trees rather than relying entirely on machine learning libraries.

Instead of using a prebuilt classifier, the algorithm implements the core splitting, tree-building, and prediction processes manually.

**Built for learning and experimenting with Machine Learning algorithms from scratch.**
