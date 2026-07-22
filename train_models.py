from pathlib import Path

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score


base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "ml_abt_service_5000_rows.csv"


data = pd.read_csv(csv_path)
data = data[data["central_block_flag"] != 1]

raw = data.copy()

# Save min/max once, from all the data. daily_scoring only sees
# customers due within 14 days, so it can't work these out itself.
scalers = {
    "days_min":  float(raw["days_until_due"].min()),
    "days_max":  float(raw["days_until_due"].max()),
    "money_min": float(raw["avg_invoice_amt_qar_12m"].min()),
    "money_max": float(raw["avg_invoice_amt_qar_12m"].max()),
}
joblib.dump(scalers, str(base_dir / "scalers.pkl"))

customer_ids = data["customer_id"]

data = data.drop(["customer_id", "vin", "sf_account_id", "businessId"], axis=1)
data = pd.get_dummies(data)

X = data.drop(["target_booked_within_14d", "target_churn_180d"], axis=1)
yb_booking = data["target_booked_within_14d"]
yc_churn = data["target_churn_180d"]

X_train, X_test, yb_train, yb_test, yc_train, yc_test, _, raw_test, _, ids_test = train_test_split(
    X,
    yb_booking,
    yc_churn,
    raw,
    customer_ids,
    test_size=0.2,
    random_state=42,
    stratify=yc_churn,
)

booking_model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
churn_model   = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)

booking_model.fit(X_train, yb_train)
churn_model.fit(X_train, yc_train)

booking_proba = booking_model.predict_proba(X_test)[:, 1]
churn_proba   = churn_model.predict_proba(X_test)[:, 1]

# Accuracy is a fake score here. Only 14% churn, so a model that always
# says "no churn" gets 86%. PR-AUC only measures finding the churners.
print("Churn base rate =", round(yc_test.mean(), 3)) # what % of customers churned
print("Churn PR-AUC =", round(average_precision_score(yc_test, churn_proba), 3)) # how good my model is at finding churners (higher = better)
print("Booking base rate =", round(yb_test.mean(), 3)) # what % of customers booked
print("Booking PR-AUC =", round(average_precision_score(yb_test, booking_proba), 3)) # how good my model is at finding bookers (higher = better)

joblib.dump(booking_model, str(base_dir / "booking_model.pkl"))
joblib.dump(churn_model, str(base_dir / "churn_model.pkl"))
joblib.dump(list(X.columns), str(base_dir / "model_columns.pkl"))
print("Saved models")