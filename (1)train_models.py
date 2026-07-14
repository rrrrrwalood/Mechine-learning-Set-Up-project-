import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score

data = pd.read_csv("ml_abt_service_5000_rows.csv")
data = data[data["central_block_flag"] != 1]

raw = data.copy()
customer_ids = data["customer_id"]

data = data.drop(["customer_id", "vin", "sf_account_id", "businessId"], axis=1)
data = pd.get_dummies(data)

X = data.drop(["target_booked_within_14d", "target_churn_180d"], axis=1)
yb_booking = data["target_booked_within_14d"]
yc_churn = data["target_churn_180d"]


# NEW ----------------------------------------------------------
# Freeze the scaling bounds here and save them.
# Before: weekly_scoring calculated min/max from whoever was in that
# week's file. Same customer, same data, different score every week.
# Now the bounds are fixed forever.
# --------------------------------------------------------------
scalers = {
    "days_min":  float(raw["days_until_due"].min()),
    "days_max":  float(raw["days_until_due"].max()),
    "money_min": float(raw["avg_invoice_amt_qar_12m"].min()),
    "money_max": float(raw["avg_invoice_amt_qar_12m"].max()),
}
joblib.dump(scalers, "scalers.pkl")


X_train, X_test, yb_train, yb_test, yc_train, yc_test, _, raw_test, _, ids_test = train_test_split(
    X,
    yb_booking,
    yc_churn,
    raw,
    customer_ids,
    test_size=0.2,        # CHANGED — was missing, silently defaulted to 0.25
    random_state=42,
    stratify=yc_churn,
)

# TODO — THIS SPLIT IS STILL RANDOM, WHICH IS CHEATING.
# It puts future rows into training and uses them to predict past rows.
# Real life = train on old data, score new customers.
# Tell me your date column name and I'll replace this with a time split.
# Your scores WILL drop when you do. That is correct.


booking_model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
churn_model   = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)

booking_model.fit(X_train, yb_train)
churn_model.fit(X_train, yc_train)

booking_proba = booking_model.predict_proba(X_test)[:, 1]
churn_proba   = churn_model.predict_proba(X_test)[:, 1]


# CHANGED ------------------------------------------------------
# Accuracy is a fake score here. Only ~14% churn, so a model that
# always says "no churn" scores 86%. PR-AUC is the correct metric.
# --------------------------------------------------------------
print("--- Metrics ---")
print("Churn base rate:      ", round(yc_test.mean(), 3))
print("Churn PR-AUC:         ", round(average_precision_score(yc_test, churn_proba), 3))
print("Booking base rate:    ", round(yb_test.mean(), 3))
print("Booking PR-AUC:       ", round(average_precision_score(yb_test, booking_proba), 3))
print("Churn accuracy (weak):", round(accuracy_score(yc_test, churn_model.predict(X_test)), 3))


# CHANGED ------------------------------------------------------
# Use the frozen scalers, not this batch's min/max.
# The 1e-9 guards against divide-by-zero.
# clip(0,1) stops new data outside the training range going negative.
# --------------------------------------------------------------
days  = raw_test["days_until_due"]
money = raw_test["avg_invoice_amt_qar_12m"]

urgency = 1 - (days - scalers["days_min"]) / max(scalers["days_max"] - scalers["days_min"], 1e-9)
value   =     (money - scalers["money_min"]) / max(scalers["money_max"] - scalers["money_min"], 1e-9)

urgency = urgency.clip(0, 1)
value   = value.clip(0, 1)

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


# NEW ----------------------------------------------------------
# THIS IS THE NUMBER THAT SELLS THE PROJECT.
# The team can only call ~50 people. Of your top 50, how many
# actually churn? Compare to random calling.
# --------------------------------------------------------------
K = 50
top_k = results.head(K)
base_rate = yc_test.mean()
precision_at_k = top_k["churn_real"].mean()

print(f"\n--- Business impact (top {K}) ---")
print(f"Random {K} calls reach     ~{base_rate * K:.0f} churners")
print(f"My top {K} calls reach     ~{precision_at_k * K:.0f} churners")
print(f"Uplift:                     {precision_at_k / base_rate:.1f}x")


# NEW ----------------------------------------------------------
# Managers don't care about probabilities. They want to know WHY.
# --------------------------------------------------------------
importances = pd.Series(churn_model.feature_importances_, index=X.columns)
print("\n--- Top 15 churn drivers ---")
print(importances.sort_values(ascending=False).head(15))


joblib.dump(booking_model, "booking_model.pkl")
joblib.dump(churn_model, "churn_model.pkl")
joblib.dump(list(X.columns), "model_columns.pkl")
print("\nSaved models + scalers")