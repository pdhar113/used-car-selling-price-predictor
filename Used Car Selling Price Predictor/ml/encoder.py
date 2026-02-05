import pandas as pd

# -----------------------------
# Load dataset
# -----------------------------
input_file = "orig-dataset.csv"
output_file = "encoded-dataset.csv"

df = pd.read_csv(input_file)

print("\nOriginal Dataset Shape:", df.shape)

# -----------------------------
# Identify categorical columns
# -----------------------------
categorical_columns = df.select_dtypes(include=['object']).columns

print("\nCategorical Columns Detected:")
for col in categorical_columns:
    print("-", col)

# Dictionary to store encodings
encoding_maps = {}

# -----------------------------
# Encode categorical columns
# -----------------------------
for col in categorical_columns:
    unique_values = df[col].astype(str).unique()
    
    # Create encoding map starting from 0
    encoding_map = {value: idx for idx, value in enumerate(unique_values)}
    
    # Apply encoding
    df[col] = df[col].astype(str).map(encoding_map)
    
    # Store mapping
    encoding_maps[col] = encoding_map

# -----------------------------
# Save encoded dataset
# -----------------------------
df.to_csv(output_file, index=False)

# -----------------------------
# Print encoding mappings
# -----------------------------
print("\nENCODING MAPPINGS (Column-wise):\n")

for col, mapping in encoding_maps.items():
    print(f"Column: {col}")
    for k, v in mapping.items():
        print(f"  {k} -> {v}")
    print("-" * 40)

print("\nEncoded dataset saved as:", output_file)
print("Encoded Dataset Shape:", df.shape)
