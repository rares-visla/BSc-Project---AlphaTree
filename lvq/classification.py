import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklvq.models import GMLVQ
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../nodes_features.csv")

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Features and labels
X = df[['area', 'compactness', 'avgRed', 'avgGreen', 'avgBlue']].values.astype('float64')
y = df['label'].to_numpy()

feature_names = ['area', 'compactness', 'avgRed', 'avgGreen', 'avgBlue']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize GMLVQ model
gmlvq = GMLVQ(
    distance_type="adaptive-squared-euclidean",
    activation_type="swish",
    activation_params={"beta": 2},
    solver_type="waypoint-gradient-descent",
    solver_params={"max_runs": 100, "k": 5, "step_size": np.array([0.1, 0.05])},
    random_state=1428,
)

# Train the model
gmlvq.fit(X_train, y_train)

# Predictions
y_pred = gmlvq.predict(X_test)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Inspect relevance matrix (feature importance)
print("\nRelevance Matrix:")
print(gmlvq.lambda_)

# The relevance matrix is available after fitting the model.
relevance_matrix = gmlvq.lambda_

# # Plot the diagonal of the relevance matrix
# fig, ax = plt.subplots()
# fig.suptitle("Relevance Matrix Diagonal")
# ax.bar(feature_names, np.diagonal(relevance_matrix))
# ax.set_ylabel("Weight")
# ax.grid(False)
#  #show the plot
# plt.savefig("relevance_matrix_diagonal.png")

