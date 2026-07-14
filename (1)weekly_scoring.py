import pandas as pd
import joblib
import os
from datetime import date

booking_model = joblib.load("booking_model.pkl")
churn_model   = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")
scalers       = joblib.load("scalers.pkl")   # NEW — the frozen bounds

data = pd.read_csv("ml_abt_service_5000_rows_NEW_FAKE.csv")
data = data[data["central_block_flag"] != 1]
data = data[data["days_until_due"].between(1, 3)].copy()

if len(data) == 0:                            # NEW — nothing to score today
    print("No customers due in 1-3 days. Nothing to do.")
    raise SystemExit

raw = data.copy()

data = data.drop(["customer_id", "vin", "sf_account_id", "businessId"], axis=1, errors="ignore")
data = pd.get_dummies(data)
data = data.drop(["target_booked_within_14d", "target_churn_180d"], axis=1, errors="ignore")


# NEW ----------------------------------------------------------
# reindex() silently fills missing columns with 0. If the new CSV
# has different category names, EVERY column becomes 0 and you score
# 5000 customers on an empty matrix — with no error at all.
# Check first, and stop if it looks broken.
# --------------------------------------------------------------
missing = set(model_columns) - set(data.columns)
extra   = set(data.columns) - set(model_columns)
print(f"Missing columns: {len(missing)} / {len(model_columns)}")
print(f"Unexpected columns (dropped): {len(extra)}")

if len(missing) > 0.3 * len(model_columns):
    raise SystemExit("Too many missing columns. New CSV doesn't match training data.")

data = data.reindex(columns=model_columns, fill_value=0)


booking_proba = booking_model.predict_proba(data)[:, 1]
churn_proba   = churn_model.predict_proba(data)[:, 1]


# CHANGED ------------------------------------------------------
# THE MAIN BUG FIX.
# Was: days.min() / days.max() from THIS WEEK'S rows.
# Because days_until_due is filtered to 1-3, if everyone that week
# has days=2 then max-min = 0 --> divide by zero --> all NaN.
# And Ahmed's score changed every week just because OTHER customers
# in the file changed.
# Now: fixed bounds from training. Ahmed's score only moves if Ahmed moves.
# --------------------------------------------------------------
days  = raw["days_until_due"]
money = raw["avg_invoice_amt_qar_12m"]

urgency = 1 - (days - scalers["days_min"]) / max(scalers["days_max"] - scalers["days_min"], 1e-9)
value   =     (money - scalers["money_min"]) / max(scalers["money_max"] - scalers["money_min"], 1e-9)

urgency = urgency.clip(0, 1)
value   = value.clip(0, 1)

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

results["date_scored"]    = date.today().isoformat()
results["contact_result"] = pd.NA
results["actual_booking"] = pd.NA
results["actual_churn"]   = pd.NA

results.to_csv("brevo_leads.csv", index=False)
print(f"Saved brevo_leads.csv ({len(results)} customers)")
print(results.head())


diary_batch = results[[
    "customer_id",
    "booking_probability",
    "churn_probability",
    "rank_score",
    "date_scored",
    "contact_result",
    "actual_booking",
    "actual_churn",
]].copy()

diary_file = "customer_diary.csv"

if os.path.exists(diary_file):
    # NEW ------------------------------------------------------
    # If you run this script twice in one day, the old code appended
    # the same customers again. Now: remove today's rows first, then
    # write. Re-running is safe.
    # ----------------------------------------------------------
    diary = pd.read_csv(diary_file)
    today = date.today().isoformat()
    diary = diary[diary["date_scored"] != today]
    diary = pd.concat([diary, diary_batch], ignore_index=True)
    diary.to_csv(diary_file, index=False)
else:
    diary_batch.to_csv(diary_file, index=False)

print("Saved customer diary")