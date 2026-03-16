import pandas as pd

# Load cleaned dataset
df = pd.read_csv("scripts/dataclean/orders_cleaned.csv")

# Remove columns that should not be used for prediction
df = df.drop(columns=[
    "Row ID",
    "Order ID",
    "Customer ID",
    "Customer Name",
    "Product ID",
    "Product Name"
])

# Convert categorical columns into numerical features
df = pd.get_dummies(
    df,
    columns=["Category", "Sub-Category", "Region", "Segment", "Ship Mode"],
    drop_first=True
)

print("Data preprocessing completed")
print("Dataset shape:", df.shape)

from sklearn.model_selection import train_test_split

# Define features and target
X = df.drop("Sales", axis=1)
y = df["Sales"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train-Test split completed")
print("Training samples:", X_train.shape[0])