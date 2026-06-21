import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Sample Dataset
data = {
    "age":[18,22,25,30,35,40],
    "interest_music":[1,1,0,0,1,0],
    "interest_movies":[0,1,1,1,0,0],
    "recommendation":[1,1,0,0,1,0]
}

df = pd.DataFrame(data)

X = df.drop("recommendation", axis=1)
y = df["recommendation"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier()

model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")

print("Model Trained Successfully")
