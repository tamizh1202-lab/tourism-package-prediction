# for data manipulation
import pandas as pd
import sklearn
import os

# for splitting
from sklearn.model_selection import train_test_split
# for encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# for Hugging Face Hub
from huggingface_hub import HfApi

# Initialize HF API with token from environment
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define dataset path from Hugging Face repo
DATASET_PATH = "hf://datasets/tamizh1296/tourism-package-prediction/tourism.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unique identifier
df.drop(columns=['CustomerID'], inplace=True)

# Define target
target_col = "ProdTaken"

# Split features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Save locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data splits saved locally.")

# Upload to Hugging Face dataset repo
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,  # just filename
        repo_id="tamizh1296/tourism-package-prediction",
        repo_type="dataset",
    )
    print(f"Uploaded {file_path} to HF dataset repo.")
