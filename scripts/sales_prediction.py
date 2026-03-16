import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load cleaned dataset
df = pd.read_csv("scripts/dataclean/orders_cleaned.csv")

# Remove identifiers (not useful for prediction)
df = df.drop(columns=[
    "Row ID",
    "Order ID",
    "Customer ID",
    "Customer Name",
    "Product ID",
    "Product Name"
])

# Convert categorical variables into numeric
df = pd.get_dummies(df, drop_first=True)

# Define features and target
X = df.drop("Sales", axis=1)
y = df["Sales"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

print("Model training completed")

from sklearn.metrics import mean_squared_error, r2_score

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model Evaluation")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)