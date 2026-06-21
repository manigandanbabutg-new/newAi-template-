import joblib

model = joblib.load("model.pkl")

def predict(user_data):

    result = model.predict([user_data])

    if result[0] == 1:
        return "Recommended"
    else:
        return "Not Recommended"

# Example Input

user = [24, 1, 1]

output = predict(user)

print(output)
