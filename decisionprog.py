import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt

# Load the dataset from your local system
file_path = r"C:\Users\T.Haneesh\desicion\1) iris.csv"
df = pd.read_csv(file_path)

# Split the data into features (X) and target (y)
X = df.drop('species', axis=1)  # Replace 'species' if your target column has a different name
y = df['species']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an unpruned decision tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Visualize the unpruned decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=clf.classes_)
plt.title("Unpruned Decision Tree")
plt.show()

# Evaluate the unpruned tree
y_pred = clf.predict(X_test)
print("Unpruned Accuracy:", accuracy_score(y_test, y_pred))
print("Unpruned F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("Unpruned Classification Report:\n", classification_report(y_test, y_pred))

# Train a pruned decision tree (to prevent overfitting)
clf_pruned = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_pruned.fit(X_train, y_train)

# Visualize the pruned decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf_pruned, filled=True, feature_names=X.columns, class_names=clf_pruned.classes_)
plt.title("Pruned Decision Tree (max_depth=3)")
plt.show()

# Evaluate the pruned tree
y_pred_pruned = clf_pruned.predict(X_test)
print("Pruned Accuracy:", accuracy_score(y_test, y_pred_pruned))
print("Pruned F1 Score:", f1_score(y_test, y_pred_pruned, average='weighted'))
print("Pruned Classification Report:\n", classification_report(y_test, y_pred_pruned))
