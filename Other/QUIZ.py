import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("ml_abt_service_1000_rows.csv")

customer_ids = data["customer_id"]


data = data.drop("customer_id", axis=1)

data = pd.get_dummies(data)

X = data.drop(["target_booked_within_14d", "target_churn_180d"] ,axis=1)
y = data["target_booked_within_14d"]

X_train, X_test, y_train, y_test, _ , ids_test  = train_test_split(X,y,customer_ids, random_state=42)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

prediction = model.predict(X_test)

acc = accuracy_score(y_test, prediction)
print("The Accuracy Score = ", acc)

probabilities = model.predict_proba(X_test)[:, 1]

for customer_id, prob in zip(ids_test, probabilities):
    print(f"{customer_id}: {prob*100:.0f}% likely to book")
