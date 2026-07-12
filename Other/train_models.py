import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("ml_abt_service_5000_rows.csv")

customer_ids = data["customer_id"]

# The model should not learn from random ID numbers
data = data.drop("customer_id", axis=1)

data = pd.get_dummies(data, drop_first=True)  # breaks down the text to numbers

# drop BOTH targets from X, so the model learns to predict them
X = data.drop(["target_booked_within_14d", "target_churn_180d"], axis=1)
y_booking = data["target_booked_within_14d"]   # who books soon (14d)
y_churn   = data["target_churn_180d"]          # who leaves for good (180d)

X_train, X_test, yb_train, yb_test, yc_train, yc_test, _, ids_test = train_test_split(
    X,
    y_booking,
    y_churn,
    customer_ids,
    test_size=0.2,
    random_state=42,
    stratify=y_churn
)

# TWO models — one per target
booking_model = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42)
churn_model   = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42)

# each learns ONE answer
booking_model.fit(X_train, yb_train)
churn_model.fit(X_train, yc_train)

# each predicts its own thing — predict() takes ONLY X_test
booking_preds = booking_model.predict(X_test)
churn_preds   = churn_model.predict(X_test)

# probabilities (the % likely)
booking_prob = booking_model.predict_proba(X_test)[:, 1]
churn_prob   = churn_model.predict_proba(X_test)[:, 1]

# accuracy of each model
print("Booking accuracy:", accuracy_score(yb_test, booking_preds))
print("Churn accuracy:  ", accuracy_score(yc_test, churn_preds), "\n")

# show first 3 customers
for customer_id, b_prob, c_prob in zip(ids_test.iloc[:3], booking_prob[:3], churn_prob[:3]):
    print(f"{customer_id}: {b_prob*100:.0f}% likely to book | {c_prob*100:.0f}% churn risk")

# build the results table
results = pd.DataFrame({
    "customer_id": ids_test.values, # throws the index away
    "booking_probability": booking_prob,
    "churn_probability": churn_prob,
    "booking_real": yb_test.values,
    "churn_real": yc_test.values,
})

results.sort_values("booking_probability", ascending=False).to_csv("hot_leads.csv", index=False)
print("\nSaved hot_leads.csv") 

joblib.dump(booking_model, "booking_model.pkl")
joblib.dump(churn_model, "churn_model.pkl")
joblib.dump(list(X.columns), "model_columns.pkl")
print("Saved: booking_model.pkl, churn_model.pkl, model_columns.pkl")