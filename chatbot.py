import aiml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("C:/Users/Rajesh/real_estate/indian-real-estate-market-dataset-2023-25/indian_realestate_dataset_1000.csv")

# Features
X = df[["built_up_area_sqft", "bedrooms", "city"]]
y = df["price_inr"]
X = pd.get_dummies(X, columns=["city"], drop_first=True)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

def predict_price(area, rooms, location):
    input_data = pd.DataFrame([[area, rooms, location]], 
                              columns=["built_up_area_sqft","bedrooms","city"])
    input_data = pd.get_dummies(input_data, columns=["city"])
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    return model.predict(input_data)[0]

# AIML Kernel
kernel = aiml.Kernel()
kernel.learn("realestate.aiml")

print("Chatbot ready! Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    
    response = kernel.respond(user_input.upper())  # AIML expects uppercase
    print("Bot:", response)
    
    if user_input.lower().startswith("predict"):
        # Example: "predict 1200 3 Pune"
        parts = user_input.split()
        if len(parts) == 4:
            area = int(parts[1])
            rooms = int(parts[2])
            city = parts[3]
            price = predict_price(area, rooms, city)
            print("Bot: Predicted Price:", round(price, 2))