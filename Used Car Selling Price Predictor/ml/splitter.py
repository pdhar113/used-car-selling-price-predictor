import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("encoded-dataset.csv")   # replace with your file name

# (Optional but recommended) Remove duplicates
df = df.drop_duplicates()

# Identify target variable
target_column = "selling_price"   # change as per your dataset

# Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Split the data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,      # 30% for testing
    random_state=42,     # reproducibility
    shuffle=True         # avoid order bias
)

# Combine X and y back (optional, useful for saving CSVs)
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Save to CSV files
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Training data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)