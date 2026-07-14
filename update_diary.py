import pandas as pd
import os
import glob

diary_file = "customer_diary.csv"

if not os.path.exists(diary_file):
    print("customer_diary.csv was not found")
    raise SystemExit

# FIX 4 — read EVERY file the CRM has returned, not just one.
# A file returned 3 weeks late still gets picked up.
return_files = sorted(glob.glob("returns/*.csv"))

if len(return_files) == 0:
    print("No files in returns/ yet")
    raise SystemExit

columns_needed = [
    "customer_id",
    "date_scored",
    "contact_result",
    "actual_booking",
    "actual_churn"
]

all_returns = []
for f in return_files:
    df = pd.read_csv(f)
    for column in columns_needed:
        if column not in df.columns:
            print(f"Skipping {f}, missing column: {column}")
            break
    else:
        all_returns.append(df[columns_needed])
        print(f"Read {f}")

if len(all_returns) == 0:
    print("No usable files in returns/")
    raise SystemExit

crm_results = pd.concat(all_returns, ignore_index=True)
crm_results = crm_results.drop_duplicates(subset=["customer_id", "date_scored"], keep="last")

diary = pd.read_csv(diary_file)

diary = diary.merge(
    crm_results,
    on=["customer_id", "date_scored"],
    how="left",
    suffixes=("", "_updated")
)

diary["contact_result"] = diary["contact_result_updated"].combine_first(diary["contact_result"])
diary["actual_booking"] = diary["actual_booking_updated"].combine_first(diary["actual_booking"])
diary["actual_churn"] = diary["actual_churn_updated"].combine_first(diary["actual_churn"])

diary = diary.drop([
    "contact_result_updated",
    "actual_booking_updated",
    "actual_churn_updated"
], axis=1)

# FIX 5 — stops duplicates piling up over months
diary = diary.drop_duplicates(subset=["customer_id", "date_scored"], keep="last")

diary.to_csv(diary_file, index=False)
print("Updated customer_diary.csv")