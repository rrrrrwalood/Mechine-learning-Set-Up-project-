import pandas as pd 

from sklearn.model_selection import train_test_split

# from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

data = pd.read_csv("ml_abt_service_5000_rows.csv")
# print(data.head())

customer_ids = data["customer_id"]

# The model should not learn from random ID numbers
data = data.drop("customer_id", axis=1)


data = pd.get_dummies(data, drop_first=True) # breaks down the text to numbers 
# print(data.head())


X = data.drop("target_booked_within_14d",  axis=1) # drop the target variables, so the model can learn to predict it
y = data["target_booked_within_14d"] # target variable

X_train, X_test, y_train, y_test, _, ids_test = train_test_split(X, y, customer_ids, test_size=0.2, random_state= 42, stratify=y)

model = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42)

model.fit(X_train, y_train) # train the model — it learns the pattern from the training clues (X_train) and their real answers (y_train)


predictions = model.predict(X_test) # use the trained model to guess answers for the hidden test customers (X_test)

matrix = confusion_matrix(y_test, predictions)
print("matrix:")
print(matrix, "\n")

#[[TN FP]
#[FN TP]]
#TN = model said no, and they really did not book
#FP = model said yes, but they did not book
#FN = model said no, but they actually booked
#TP = model said yes, and they actually booked


acc = accuracy_score(y_test, predictions)
# compare the model's guesses against the real answers, get the % it got right
print("The Accuracy: ", acc, "\n") 


results = X_test.copy()

results["prediction"] = model.predict_proba(X_test)[:, 1]
results["real_answer"] = y_test               # what actually happened (0 or 1)
results["correct"] = (predictions == y_test)  # True if model got it right

probabilities = model.predict_proba(X_test)[:, 1]

results = pd.DataFrame({
    "customer_id": ids_test.values,
    "prediction_probability": probabilities,
    "real_answer": y_test.values,
    "correct": predictions == y_test.values
})

for customer_id, probability in zip(ids_test.iloc[:3], probabilities[:3]):
    print(f"{customer_id}: {probability * 100:.0f}% likely to book")
    
print(classification_report(y_test, predictions))