import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Reduce to 2 features using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2)

# SVM model
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

# Accuracy
acc = model.score(X_test, y_test)
print("Accuracy:", acc)

# Plot decision boundary
plt.scatter(X_pca[:,0], X_pca[:,1], c=y)
plt.title("SVM with PCA")
plt.show()

# GridSearch for best params
param_grid = {'C':[0.1,1,10], 'gamma':[0.1,1,10]}
grid = GridSearchCV(SVC(), param_grid)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)