import pandas as pd
import joblib

booking_model = joblib.load("booking_model.pkl")
churn_model   = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

data = pd.read_csv("ml_abt_service_5000_rows.csv")
raw = data.copy()

data = data.drop("customer_id", axis=1)
data = pd.get_dummies(data, drop_first=True)
data = data.drop(["target_booked_within_14d", "target_churn_180d"], axis=1, errors="ignore")
data = data.reindex(columns=model_columns, fill_value=0) 

booking_prob = booking_model.predict_proba(data)[:, 1]
churn_prob   = churn_model.predict_proba(data)[:, 1]

# pull urgency + value from the clean copy
days_until_due = raw["days_until_due"]
invoice_value  = raw["avg_invoice_amt_qar_12m"]

# squash both to 0–1 so they mix fairly
urgency = 1 - (days_until_due - days_until_due.min()) / (days_until_due.max() - days_until_due.min())
value   = (invoice_value - invoice_value.min()) / (invoice_value.max() - invoice_value.min())

# blend all 4 into one score
rank_score = (0.4 * churn_prob) + (0.3 * booking_prob) + (0.2 * urgency) + (0.1 * value)

# build the final table for the outreach team
output = pd.DataFrame({
    "customer_id": raw["customer_id"],
    "rank_score": rank_score,
    "churn_probability": churn_prob,
    "booking_probability": booking_prob,
    "days_until_due": raw["days_until_due"],
    "preferred_channel": raw["preferred_channel"],
    "whatsapp_optin": raw["whatsapp_optin"],
    "call_optin": raw["call_optin"],
    # "preferred_contact_hour": raw["preferred_contact_hour"],
    # "last_service_branch": raw["last_service_branch"],
})

# best leads at the top
output = output.sort_values("rank_score", ascending=False)

# save it
output.to_csv("hot_leads.csv", index=False)
print("Saved hot_leads.csv — top customers first")
print(output.head(10))