import pandas as pd
import joblib
import os
from datetime import date

booking_model = joblib.load("booking_model.pkl")
churn_model   = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")
scalers       = joblib.load("scalers.pkl")   # FIX 1

today = date.today().isoformat()

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

# FIX 1 — use the saved min/max, not today's.
# Before: days.min()/max() came from today's rows. Since days is filtered
# to 1-3, if everyone today had days=2 then max-min=0 -> divide by zero.
# And a customer's score changed daily just because OTHER customers changed.
days  = raw["days_until_due"]
money = raw["avg_invoice_amt_qar_12m"]

urgency = 1 - (days - scalers["days_min"]) / (scalers["days_max"] - scalers["days_min"])
value   = (money - scalers["money_min"]) / (scalers["money_max"] - scalers["money_min"])

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

results["date_scored"]    = today
results["contact_result"] = pd.NA
results["actual_booking"] = pd.NA
results["actual_churn"]   = pd.NA

# FIX 4 — this runs EVERY DAY. brevo_leads.csv was being overwritten
# each morning, destroying anything the CRM hadn't returned yet.
# Now every day gets its own file.
os.makedirs("leads", exist_ok=True)
leads_file = f"leads/leads_{today}.csv"
results.to_csv(leads_file, index=False)
print(f"Saved {leads_file}")
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
    # FIX 5 — if you re-run this today, delete today's rows first.
    # The old code appended them again and duplicated everything.
    diary = pd.read_csv(diary_file)
    diary = diary[diary["date_scored"] != today]
    diary = pd.concat([diary, diary_batch], ignore_index=True)
    diary.to_csv(diary_file, index=False)
else:
    diary_batch.to_csv(diary_file, index=False)

print("Saved customer diary")