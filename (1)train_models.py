import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score

data = pd.read_csv("ml_abt_service_5000_rows.csv")
data = data[data["central_block_flag"] != 1]

raw = data.copy()

# FIX 1 — save the min/max once, from all the data.
# weekly_scoring only sees people due in 1-3 days, so it can't
# work these out itself. It loads them from here.
scalers = {
    "days_min":  float(raw["days_until_due"].min()),
    "days_max":  float(raw["days_until_due"].max()),
    "money_min": float(raw["avg_invoice_amt_qar_12m"].min()),
    "money_max": float(raw["avg_invoice_amt_qar_12m"].max()),
}
joblib.dump(scalers, "scalers.pkl")

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
    test_size=0.2,        # FIX 2 — was missing, silently used 0.25
    random_state=42,
    stratify=yc_churn,
)

booking_model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
churn_model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)

booking_model.fit(X_train, yb_train)
churn_model.fit(X_train, yc_train)

booking_proba = booking_model.predict_proba(X_test)[:, 1]
churn_proba = churn_model.predict_proba(X_test)[:, 1]

# FIX 3 — accuracy is a fake score.
# Only 14% churn, so a model that always says "no churn" gets 86%.
# PR-AUC only measures how well you find the churners.
print("Churn base rate = ", round(yc_test.mean(), 3))
print("Churn PR-AUC = ", round(average_precision_score(yc_test, churn_proba), 3))
print("Booking base rate = ", round(yb_test.mean(), 3))
print("Booking PR-AUC = ", round(average_precision_score(yb_test, booking_proba), 3))

days = raw_test["days_until_due"]
money = raw_test["avg_invoice_amt_qar_12m"]

urgency = 1 - (days - scalers["days_min"]) / (scalers["days_max"] - scalers["days_min"])
value = (money - scalers["money_min"]) / (scalers["money_max"] - scalers["money_min"])

rank_score = (0.4 * churn_proba) + (0.3 * booking_proba) + (0.2 * urgency.values) + (0.1 * value.values)

results = pd.DataFrame({
    "customer_id": ids_test.values,
    "rank_score": rank_score,
    "booking_probability": booking_proba,
    "churn_probability": churn_proba,
    "booking_real": yb_test.values,
    "churn_real": yc_test.values,
    "preferred_channel": raw_test["preferred_channel"].values,
    "whatsapp_optin": raw_test["whatsapp_optin"].values,
    "call_optin": raw_test["call_optin"].values,
    "preferred_contact_hour": raw_test["preferred_contact_hour"].values,
    "days_until_due": raw_test["days_until_due"].values,
    "last_service_branch": raw_test["last_service_branch"].values,
})

results = results.sort_values("rank_score", ascending=False)

joblib.dump(booking_model, "booking_model.pkl")
joblib.dump(churn_model, "churn_model.pkl")
joblib.dump(list(X.columns), "model_columns.pkl")
print("Saved models")