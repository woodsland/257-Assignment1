# COMP257 - Unsupervised & Reinforcement Learning (Section 002)
# Assignment 1 - PCA (Question 1)
# Name: Wai Lim Leung
# ID  : 301276989
# Date: 18-Sep-2023

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, IncrementalPCA
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

# 1.1 Load the MNIST dataset (70,000 instances and 784 columns).
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data
y = mnist.target.astype(int)
print("1.1 Mnist Set:")
print(mnist.data.shape)

# 1.2 Show 10 values
def plot_digits(data, num_samples=10):
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
    for i in range(num_samples):
        axes[i].imshow(data[i].reshape(28, 28), cmap="gray")
        axes[i].axis("off")
    plt.show()

plot_digits(X[:10])

# Split the dataset into training & testing data - First 60,000 instance)
X_train, y_train = mnist.data[:60000], mnist.target[:60000]
print(X_train.shape)

# Remaining 10,000 instances
X_test, y_test = mnist.data[60000:], mnist.target[60000:]
print(X_test.shape)

# 1.3 Retrieve first & second principal component
X_Centered = X_train - X_train.mean(axis=0)
U, s, Vt = np.linalg.svd(X_Centered)
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]

pca = PCA(n_components=2)
pca.fit_transform(X_train)
print()
print("1.3 First & Second Principal Component & Ratio")
print(pca.explained_variance_ratio_)

# 1.4 Plot projection
project_1 = np.dot(X_Centered, pca.components_[0])
project_2 = np.dot(X_Centered, pca.components_[1])

plt.figure(figsize=(8, 4))
plt.subplot(3, 1, 1)
plt.scatter(project_1, np.zeros_like(project_1), alpha=0.5)
plt.title("1st Principal Component")
plt.xlabel("Projection")
plt.yticks([])

plt.subplot(3, 1, 3)
plt.scatter(project_2, np.zeros_like(project_2), alpha=0.5)
plt.title("2nd Principal Component")
plt.xlabel("Projection")
plt.yticks([])

plt.tight_layout()
plt.show()

# 1.5 Incremental PCA
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_Centered in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_Centered)

X_reduced = inc_pca.transform(X_train)

# 1.6 Display Original and Compressed Set
original_digits = X_Centered[:10]
plot_digits(original_digits)

compressed_digits = inc_pca.inverse_transform(X_reduced[:10])
plot_digits(compressed_digits)