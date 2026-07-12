import pandas as pd
import joblib
import os
from datetime import date

booking_model = joblib.load("booking_model.pkl")
churn_model   = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

data = pd.read_csv("ml_abt_service_5000_rows_NEW_FAKE.csv")
data = data[data["central_block_flag"] != 1]
data = data[data["days_until_due"].between(1, 3)].copy()

raw = data.copy()

data = data.drop(["customer_id", "vin", "sf_account_id", "businessId"], axis=1, errors="ignore")
data = pd.get_dummies(data)
data = data.drop(["target_booked_within_14d", "target_churn_180d"], axis=1, errors="ignore")
data = data.reindex(columns=model_columns, fill_value=0)

booking_proba = booking_model.predict_proba(data)[:, 1]
churn_proba   = churn_model.predict_proba(data)[:, 1]

days = raw["days_until_due"]
money = raw["avg_invoice_amt_qar_12m"]

urgency = 1 - (days - days.min()) / (days.max() - days.min())
value = (money - money.min()) / (money.max() - money.min())

rank_score = (0.4 * churn_proba) + (0.3 * booking_proba) + (0.2 * urgency.values) + (0.1 * value.values)

results = pd.DataFrame({
    "customer_id": raw["customer_id"].values,
    "rank_score": rank_score,
    "booking_probability": booking_proba,
    "churn_probability": churn_proba,
    "preferred_channel": raw["preferred_channel"].values,
    "whatsapp_optin": raw["whatsapp_optin"].values,
    "call_optin": raw["call_optin"].values,
    "preferred_contact_hour": raw["preferred_contact_hour"].values,
    "days_until_due": raw["days_until_due"].values,
    "last_service_branch": raw["last_service_branch"].values,
})

results = results.sort_values("rank_score", ascending=False)

# Add columns the CRM will fill in later
results["date_scored"] = date.today().isoformat()
results["contact_result"] = pd.NA
results["actual_booking"] = pd.NA
results["actual_churn"] = pd.NA

results.to_csv("brevo_leads.csv", index=False)
print("Saved brevo_leads.csv")
print(results.head())


diary_batch = results[[
    "customer_id",
    "booking_probability",
    "churn_probability",
    "rank_score",
    "date_scored",
    "contact_result",
    "actual_booking",
    "actual_churn"
]].copy()

diary_file = "customer_diary.csv"

if os.path.exists(diary_file):
    diary_batch.to_csv(diary_file, mode="a", header=False, index=False)
else:
    diary_batch.to_csv(diary_file, index=False)

print("Saved customer diary")













# # ---------------------------------------------------------
# # BUILD STEP 1: log this week's predictions into a growing
# # outcomes_log.csv, instead of letting hot_leads.csv get
# # overwritten and lost every week.
# # actual_booking / actual_churn are left BLANK for now.
# # Filling them in with real outcomes is BUILD STEP 2.
# # ---------------------------------------------------------

# outcomes_batch = pd.DataFrame({
#     "customer_id": raw["customer_id"].values,
#     "week_scored": date.today().isoformat(),   # e.g. "2026-07-01"
#     "predicted_booking": booking_proba,
#     "predicted_churn": churn_proba,
#     "actual_booking": pd.NA,   # blank, filled in later
#     "actual_churn": pd.NA,     # blank, filled in later
# })

# log_file = "outcomes_log.csv"

# if os.path.exists(log_file):
#     # append without writing the header again
#     outcomes_batch.to_csv(log_file, mode="a", header=False, index=False)
# else:
#     # first time running: create the file with a header
#     outcomes_batch.to_csv(log_file, mode="w", header=True, index=False)

# print(f"Logged {len(outcomes_batch)} rows into {log_file}")

