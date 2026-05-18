# Iris Flower Classification using Machine Learning

## Project Overview

This project uses multiple Machine Learning classification algorithms to classify Iris flower species based on flower measurements.

The project compares the performance of:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

The dataset is preprocessed using Label Encoding and evaluated using Accuracy Score, Confusion Matrix, and Classification Report.

---

## Technologies Used

- Python
- Pandas
- Scikit-Learn

---

## Libraries Used

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

---

## Dataset

Dataset used:
- iris.csv

The dataset contains flower measurements such as:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Target Column:
- Species

Species included:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

---

## Data Preprocessing

### Label Encoding

The Species column is converted into numerical values using LabelEncoder.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
```

Example:
- Setosa → 0
- Versicolor → 1
- Virginica → 2

---

## Project Workflow

1. Import dataset
2. Encode target labels
3. Split dataset into training and testing data
4. Train multiple Machine Learning models
5. Predict species
6. Evaluate model performance
7. Compare results of different algorithms

---

## Machine Learning Models Used

### Logistic Regression

A supervised Machine Learning algorithm used for classification problems.

```python
lr = LogisticRegression(max_iter=1000)
```

---

### K-Nearest Neighbors (KNN)

A distance-based classification algorithm.

```python
knn = KNeighborsClassifier(n_neighbors=5)
```

---

### Gaussian Naive Bayes

A probabilistic classification algorithm based on Bayes theorem.

```python
nb = GaussianNB()
```

---

## Model Evaluation

The models are evaluated using:
- Accuracy Score
- Confusion Matrix
- Classification Report

```python
print("Accuracy:", accuracy_score(y, y_pred))
print("Confusion Matrix:", confusion_matrix(y, y_pred))
print("Classification Report:", classification_report(y, y_pred))
```

---

## Sample Output

```python
Logistic Regression
Accuracy: 1.00

KNN
Accuracy: 0.99

Naive Bayes
Accuracy: 0.99
```

In this project, Logistic Regression achieved the highest accuracy compared to KNN and Naive Bayes on the Iris dataset.

Note: Results may vary depending on train-test split and random state.

---

## Key Concepts Covered

- Classification Algorithms
- Label Encoding
- Train-Test Split
- Model Comparison
- Accuracy Evaluation
- Confusion Matrix
- Classification Report

---

## Future Improvements

- Add data visualization
- Use cross-validation
- Perform hyperparameter tuning
- Add feature scaling
- Deploy model using Flask or Streamlit

---

## Author

Shanu
# Iris-Flower-Classification-using-Machine-Learning
Iris Flower Classification project using Logistic Regression, KNN, and Naive Bayes with Python and Scikit-Learn.
