# COMP257 - Unsupervised & Reinforcement Learning (Section 002)
# Assignment 1 - PCA (Question 2)
# Name: Wai Lim Leung
# ID  : 301276989
# Date: 18-Sep-2023

from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

# 2.1 Load the MNIST dataset (Use first 10000 record only because running time).
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype(float)
y = mnist.target.astype(int)
X_mnist = X[:10000]
y_mnist = y[:10000]

# 2.2 Swiss roll
X_swiss_data, _ = make_swiss_roll(n_samples=10000, noise=0.5)
fig = plt.figure(figsize=(8, 6))
sw = fig.add_subplot(1, 1, 1, projection='3d')
sw.scatter(X_swiss_data[:, 0], X_swiss_data[:, 1], X_swiss_data[:, 2], c=X_swiss_data[:, 2], cmap=plt.cm.Spectral)
sw.set_xlabel('X')
sw.set_ylabel('Y')
sw.set_zlabel('Z')
sw.set_title('2.2 Swiss Roll')
plt.show()

# 2.3a Linear
kpca_linear = KernelPCA(kernel="linear", n_components=2)
X_kpca_linear = kpca_linear.fit_transform(X_swiss_data)

# 2.3b RBF
kpca_rbf = KernelPCA(kernel="rbf", gamma=0.04, n_components=2)
X_kpca_rbf = kpca_rbf.fit_transform(X_swiss_data)

# 2.3c Sigmoid
kpca_sigmoid = KernelPCA(kernel="sigmoid", gamma=1e-3, coef0=1, n_components=2)
X_kpca_sigmoid = kpca_sigmoid.fit_transform(X_swiss_data)

# 2.4 Plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(X_kpca_linear[:, 0], X_kpca_linear[:, 1], c=y_mnist[:10000], cmap=plt.cm.Paired)
plt.title("2.4a Linear Kernel")

plt.subplot(1, 3, 2)
plt.scatter(X_kpca_rbf[:, 0], X_kpca_rbf[:, 1], c=y_mnist[:10000], cmap=plt.cm.Paired)
plt.title("2.4b RBF Kernel")

plt.subplot(1, 3, 3)
plt.scatter(X_kpca_sigmoid[:, 0], X_kpca_sigmoid[:, 1], c=y_mnist[:10000], cmap=plt.cm.Paired)
plt.title("2.4c Sigmoid Kernel")

plt.show()

# 2.5 Best Parameters
clf = Pipeline([
    ('kpca', KernelPCA(n_components=2)),
    ('logreg', LogisticRegression())
])

param_grid = {
    'kpca__gamma': [0.03, 0.05, 10],
    'kpca__kernel': ['rbf', 'sigmoid'],
}

grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X_mnist, y_mnist)
print("2.5 Best Parameters: ", grid_search.best_params_)

X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size=0.3, random_state=42)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("2.5 Accuracy: ", accuracy)

# 2.6 Plot result using GridSearchCV
plt.figure(figsize=(10, 5))
results = grid_search.cv_results_

for i, (kernel, gamma) in enumerate(zip(results['param_kpca__kernel'], results['param_kpca__gamma'])):
    plt.subplot(2, 3, i + 1)
    plt.scatter(X_kpca_linear[:, 0], X_kpca_linear[:, 1], c=y_mnist[:10000], cmap=plt.cm.Paired)
    plt.title(f"Kernel: {kernel}, Gamma: {gamma}")

plt.tight_layout()
plt.show()