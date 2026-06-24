import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("C:/Users/Rajesh/real_estate/indian-real-estate-market-dataset-2023-25/indian_realestate_dataset_1000.csv")

X = df[["built_up_area_sqft","bedrooms","bathrooms","age_years","city"]]
y = df["price_inr"]
X = pd.get_dummies(X, columns=["city"], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

def predict_price(area, rooms, bathrooms, age, location):
    input_data = pd.DataFrame([[area, rooms, bathrooms, age, location]], 
                              columns=["built_up_area_sqft","bedrooms","bathrooms","age_years","city"])
    input_data = pd.get_dummies(input_data, columns=["city"])
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    return model.predict(input_data)[0]

print("Model ready. Type values to predict:")

while True:
    user = input("Enter: area bedrooms bathrooms age city (or 'exit'): ")
    if user.lower() == "exit":
        break
    parts = user.split()
    if len(parts) == 5:
        area = int(parts[0]); rooms = int(parts[1]); baths = int(parts[2]); age = int(parts[3]); city = parts[4]
        price = predict_price(area, rooms, baths, age, city)
        print("Predicted Price (INR):", round(price, 2))
    else:
        print("Format error. Example: 1200 3 2 5 Pune")
