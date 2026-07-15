import pandas as pd
import os
import glob

diary_file = "customer_diary.csv"

# Stop if there's no diary yet
if not os.path.exists(diary_file):
    print("customer_diary.csv was not found")
    raise SystemExit

# Grab every file the CRM has sent back (even old ones)
return_files = sorted(glob.glob("returns/*.csv"))

if len(return_files) == 0:
    print("No files in returns/ yet")
    raise SystemExit

# The columns we need from each CRM file
columns_needed = [
    "customer_id",
    "date_scored",
    "contact_result",
    "actual_booking",
    "actual_churn"
]

# Read all the return files into one big list
all_returns = []
for f in return_files:
    df = pd.read_csv(f)
    for column in columns_needed:
        if column not in df.columns:       # skip any file missing a column
            print(f"Skipping {f}, missing column: {column}")
            break
    else:
        all_returns.append(df[columns_needed])
        print(f"Read {f}")

if len(all_returns) == 0:
    print("No usable files in returns/")
    raise SystemExit

# Stack all the returns together into one table
crm_results = pd.concat(all_returns, ignore_index=True)
# If the same customer+date shows up twice, keep the newest
crm_results = crm_results.drop_duplicates(subset=["customer_id", "date_scored"], keep="last")

# Open the diary
diary = pd.read_csv(diary_file)

# Match the CRM answers onto the diary by customer + date
diary = diary.merge(
    crm_results,
    on=["customer_id", "date_scored"],
    how="left",
    suffixes=("", "_updated")
)

# Fill in the real outcomes from the CRM (keep old value if there's no new one)
diary["contact_result"] = diary["contact_result_updated"].combine_first(diary["contact_result"])
diary["actual_booking"] = diary["actual_booking_updated"].combine_first(diary["actual_booking"])
diary["actual_churn"]   = diary["actual_churn_updated"].combine_first(diary["actual_churn"])

# Remove the leftover helper columns from the merge
diary = diary.drop([
    "contact_result_updated",
    "actual_booking_updated",
    "actual_churn_updated"
], axis=1)

# Safety: no duplicate rows for the same customer+date
diary = diary.drop_duplicates(subset=["customer_id", "date_scored"], keep="last")

# Save the updated diary
diary.to_csv(diary_file, index=False)

print("Updated customer_diary.csv")
print(f"{len(diary)} rows, {diary['actual_churn'].notna().sum()} with confirmed outcomes")