import pandas as pd
import joblib
import os
from datetime import date

booking_model = joblib.load("booking_model.pkl")
churn_model   = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")
scalers       = joblib.load("scalers.pkl")

today = date.today().isoformat()

data = pd.read_csv("ml_abt_service_5000_rows_NEW_FAKE.csv")
data = data[data["central_block_flag"] != 1]

# B1 — was between(1, 3). The business rule says outreach starts at N-7,
# so 1-3 days was far too late — the whole digital sequence was missed.
# N-14 to N-0 is the real window.
data = data[data["days_until_due"] <= 14].copy()

# B2 — SELF-CLEANSING.
# Don't contact someone who already booked or responded. Check the diary first.

already_handled = set()# sitcky note # Get the list of customers we've already contacted (from the diary), new skip
if os.path.exists("customer_diary.csv"):
    diary = pd.read_csv("customer_diary.csv") 
    handled = diary[diary["contact_result"].notna()] # rows the CRM already replied to
    already_handled = set(handled["customer_id"].values) #You take those contacted people's IDs and write them all on your sticky note.



before = len(data) # Remove those already-contacted customers from today's list
data = data[~data["customer_id"].isin(already_handled)] # already contacted customers are removed from today's list, so we don't contact them again. This is the self-cleansing step.
print(f"Excluded {before - len(data)} customers who already responded")


if len(data) == 0: # If nobody is left to call, stop here instead of crashing later
    print("Nobody to contact today.")
    raise SystemExit

raw = data.copy() # Keep a clean copy before we chop the data up for the model


data = data.drop(["customer_id", "vin", "sf_account_id", "businessId"], axis=1, errors="ignore")
data = pd.get_dummies(data)
data = data.drop(["target_booked_within_14d", "target_churn_180d"], axis=1, errors="ignore")
data = data.reindex(columns=model_columns, fill_value=0)

booking_proba = booking_model.predict_proba(data)[:, 1]
churn_proba   = churn_model.predict_proba(data)[:, 1]

# Use the SAVED min/max, not today's. Before, a customer's score changed
# day to day just because other customers were in the file.
days  = raw["days_until_due"]
money = raw["avg_invoice_amt_qar_12m"]

urgency = 1 - (days - scalers["days_min"]) / (scalers["days_max"] - scalers["days_min"])
value   = (money - scalers["money_min"]) / (scalers["money_max"] - scalers["money_min"])

urgency = urgency.clip(0, 1)
value   = value.clip(0, 1)

rank_score = (0.4 * churn_proba) + (0.3 * booking_proba) + (0.2 * urgency.values) + (0.1 * value.values)

results = pd.DataFrame({
    "customer_id": raw["customer_id"].values,
    "rank_score": rank_score,
    "booking_probability": booking_proba,
    "churn_probability": churn_proba,
    "days_until_due": raw["days_until_due"].values,
    "preferred_channel": raw["preferred_channel"].values,
    "whatsapp_optin": raw["whatsapp_optin"].values,
    "sms_optin": raw["sms_optin"].values,
    "email_optin": raw["email_optin"].values,
    "call_optin": raw["call_optin"].values,
    "marketing_consent": raw["marketing_consent"].values,
    "preferred_contact_hour": raw["preferred_contact_hour"].values,
    "last_service_branch": raw["last_service_branch"].values,
})

# B1 — which reminder is this? The CRM needs to know.
def get_stage(days):
    if days >= 6:
        return "N-7 initial"
    elif days >= 4:
        return "N-5 follow-up"
    else:
        return "Call Centre"

results["outreach_stage"] = results["days_until_due"].apply(get_stage)

# B3 — WhatsApp first, SMS, then email, call centre last.
# Never message someone on a channel they didn't opt in to.
def resolve_channel(row):
    if row["whatsapp_optin"] == 1:
        return "WhatsApp"
    elif row["sms_optin"] == 1:
        return "SMS"
    elif row["email_optin"] == 1:
        return "Email"
    else:
        return "Call Centre"

results["resolved_channel"] = results.apply(
    lambda r: "Call Centre" if r["outreach_stage"] == "Call Centre" else resolve_channel(r),
    axis=1
)
results = results.sort_values("rank_score", ascending=False)

results["date_scored"]    = today
results["contact_result"] = pd.NA
results["actual_booking"] = pd.NA
results["actual_churn"]   = pd.NA

# Each day gets its own file. Before, brevo_leads.csv was overwritten
# every morning, destroying anything the CRM hadn't returned yet.
os.makedirs("leads", exist_ok=True)
leads_file = f"leads/leads_{today}.csv"
results.to_csv(leads_file, index=False)
print(f"Saved {leads_file} ({len(results)} customers)")

diary_batch = results[[
    "customer_id",
    "booking_probability",
    "churn_probability",
    "rank_score",
    "outreach_stage",
    "resolved_channel",
    "date_scored",
    "contact_result",
    "actual_booking",
    "actual_churn"
]].copy()

diary_file = "customer_diary.csv"

if os.path.exists(diary_file):
    # If you re-run today, delete today's rows first. The old code
    # appended them again and duplicated everything.
    diary = pd.read_csv(diary_file)
    diary = diary[diary["date_scored"] != today]
    diary = pd.concat([diary, diary_batch], ignore_index=True)
    diary.to_csv(diary_file, index=False)
else:
    diary_batch.to_csv(diary_file, index=False)

print("Saved customer diary")