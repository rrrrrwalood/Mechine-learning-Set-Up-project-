import pandas as pd
import os

diary_file = "customer_diary.csv"
crm_file = "brevo_leads.csv"

if not os.path.exists(diary_file):
    print("customer_diary.csv was not found")
    raise SystemExit

if not os.path.exists(crm_file):
    print("brevo_leads.csv was not found")
    raise SystemExit


diary = pd.read_csv(diary_file)
crm_results = pd.read_csv(crm_file)


columns_needed = [
    "customer_id",
    "date_scored",
    "contact_result",
    "actual_booking",
    "actual_churn"
]

for column in columns_needed:
    if column not in crm_results.columns:
        print(f"Missing column: {column}")
        raise SystemExit


diary = diary.merge(
    crm_results[columns_needed],
    on=["customer_id", "date_scored"],
    how="left",
    suffixes=("", "_updated")
)


diary["contact_result"] = diary["contact_result_updated"].combine_first(
    diary["contact_result"]
)

diary["actual_booking"] = diary["actual_booking_updated"].combine_first(
    diary["actual_booking"]
)

diary["actual_churn"] = diary["actual_churn_updated"].combine_first(
    diary["actual_churn"]
)


diary = diary.drop([
    "contact_result_updated",
    "actual_booking_updated",
    "actual_churn_updated"
], axis=1)


diary.to_csv(diary_file, index=False)

print("Updated customer_diary.csv")