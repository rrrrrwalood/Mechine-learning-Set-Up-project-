import pandas as pd 
import joblib #the tool that saves and loads your trained models.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("ml_abt_service_5000_rows.csv")

data = data[data["central_block_flag"] != 1]      

raw = data.copy()

customer_ids = data["customer_id"]

data = data.drop(["customer_id", "vin", "sf_account_id", "businessId"], axis=1)


data = pd.get_dummies(data)

X = data.drop(["target_booked_within_14d", "target_churn_180d"] ,axis=1)
yb_booking = data["target_booked_within_14d"]
yc_churn = data["target_churn_180d"]

X_train, X_test, yb_train, yb_test, yc_train, yc_test, _ , raw_test, _, ids_test = train_test_split(
    X,
    yb_booking, 
    yc_churn, 
    raw, 
    customer_ids, 
    random_state=42,
    stratify=yc_churn #we want both parts to have the same 14% churn ratio as your full data.
)


booking_model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
churn_model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)


booking_model.fit(X_train, yb_train)
churn_model.fit(X_train, yc_train)

booking_preds = booking_model.predict(X_test)
churn_preds = churn_model.predict(X_test)

booking_proba = booking_model.predict_proba(X_test)[:, 1]
churn_proba  = churn_model.predict_proba(X_test)[:, 1]

acc_yb = accuracy_score(yb_test, booking_preds)
acc_yc = accuracy_score(yc_test, churn_preds)
print("The Booking Accuracy Score = ", acc_yb)
print("The Churn Accuracy Score = ", acc_yc)


# for customer_id, b_prob, c_prob, channel in zip(ids_test.iloc[:3], booking_proba[:3], churn_proba[:3], raw_test["preferred_channel"].iloc[:3]):
#     print(f"{customer_id}: {b_prob*100:.0f}% likely to book | {c_prob*100:.0f}% churn risk | {channel}")

days = raw_test["days_until_due"]
money = raw_test["avg_invoice_amt_qar_12m"]

urgency = 1 - (days - days.min()) / (days.max() - days.min())
value = (money - money.min()) / (money.max() - money.min())

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

# results.to_csv("hot_leads.csv", index=False)
# print("Saved hot_leads.csv")
# print(results.head())


joblib.dump(booking_model, "booking_model.pkl")
joblib.dump(churn_model, "churn_model.pkl")
joblib.dump(list(X.columns), "model_columns.pkl")
print("Saved models")