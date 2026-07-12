import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load data
data = pd.read_csv("ml_abt_service_5000_rows.csv")

# Remove customer_id if it exists
if "customer_id" in data.columns:
    data = data.drop("customer_id", axis=1)

# Target
target = "target_booked_within_14d"

X = data.drop(target, axis=1)
y = data[target]

# Find number columns and text columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "bool"]).columns

# Process number columns
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Process text columns
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine processing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Model
model = RandomForestClassifier(
    n_estimators=300,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

# Full pipeline
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train
clf.fit(X_train, y_train)

# Predict
predictions = clf.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, predictions))
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print()
print("Classification Report:")
print(classification_report(y_test, predictions))

# Save model
joblib.dump(clf, "booking_prediction_model.pkl")

print("Model saved as booking_prediction_model.pkl")