import pandas as pd
import os

diary_file = "customer_diary.csv"
crm_file   = "brevo_leads.csv"

if not os.path.exists(diary_file):
    print("customer_diary.csv was not found")
    raise SystemExit

if not os.path.exists(crm_file):
    print("brevo_leads.csv was not found")
    raise SystemExit

diary       = pd.read_csv(diary_file)
crm_results = pd.read_csv(crm_file)

columns_needed = [
    "customer_id",
    "date_scored",
    "contact_result",
    "actual_booking",
    "actual_churn",
]

for column in columns_needed:
    if column not in crm_results.columns:
        print(f"Missing column: {column}")
        raise SystemExit


# NEW ----------------------------------------------------------
# If the CRM ever returns the same customer twice for the same date,
# the merge below would DUPLICATE the diary rows. Clean the CRM file first.
# --------------------------------------------------------------
crm_results = crm_results.drop_duplicates(
    subset=["customer_id", "date_scored"], keep="last"
)


diary = diary.merge(
    crm_results[columns_needed],
    on=["customer_id", "date_scored"],
    how="left",
    suffixes=("", "_updated"),
)

diary["contact_result"] = diary["contact_result_updated"].combine_first(diary["contact_result"])
diary["actual_booking"] = diary["actual_booking_updated"].combine_first(diary["actual_booking"])
diary["actual_churn"]   = diary["actual_churn_updated"].combine_first(diary["actual_churn"])

diary = diary.drop([
    "contact_result_updated",
    "actual_booking_updated",
    "actual_churn_updated",
], axis=1)


# NEW ----------------------------------------------------------
# Final safety net against duplicate rows building up over months.
# --------------------------------------------------------------
diary = diary.drop_duplicates(subset=["customer_id", "date_scored"], keep="last")

diary.to_csv(diary_file, index=False)
print(f"Updated customer_diary.csv ({len(diary)} rows)")


# NEW ----------------------------------------------------------
# CLOSING THE LOOP — this is the part nobody does.
#
# You save predictions. The CRM sends back real outcomes. But until
# now, nothing ever COMPARED them. So you never found out if the
# model was right.
#
# This groups customers by what you predicted, and shows what
# actually happened. If you predicted 70% and only 40% churned,
# your model is overconfident — and now you know.
# --------------------------------------------------------------
scored = diary.dropna(subset=["actual_churn"])

if len(scored) == 0:
    print("\nNo confirmed outcomes yet. Run again once the CRM fills them in.")
else:
    print(f"\n--- Model check ({len(scored)} confirmed outcomes) ---")

    scored = scored.copy()
    scored["predicted_bucket"] = (scored["churn_probability"] * 10).astype(int) / 10

    check = scored.groupby("predicted_bucket").agg(
        predicted=("churn_probability", "mean"),
        actual=("actual_churn", "mean"),
        customers=("actual_churn", "count"),
    ).round(2)

    print(check)
    print("\nIf 'predicted' and 'actual' are close, the model is calibrated.")
    print("If predicted is much higher, it is overconfident.")