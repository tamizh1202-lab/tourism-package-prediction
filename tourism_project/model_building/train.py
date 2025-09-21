# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for Hugging Face Hub
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import os
import mlflow

mlflow.set_tracking_uri("https://7b5f58aed2f2.ngrok-free.app")
mlflow.set_experiment("MLOps_experiment")

# Initialize HF API with token from environment
api = HfApi(token=os.getenv("HF_TOKEN"))

Xtrain_path = "hf://datasets/tamizh1296/tourist-package-prediction/tree/main/Xtrain.csv"
Xtest_path = "hf://datasets/tamizh1296/tourist-package-prediction/tree/main/Xtest.csv"
ytrain_path = "hf://datasets/tamizh1296/tourist-package-prediction/tree/main/ytrain.csv"
ytest_path = "hf://datasets/tamizh1296/tourist-package-prediction/tree/main/ytest.csv"

# Load train/test data
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest = pd.read_csv(ytest_path).squeeze()

print("Data loaded successfully.")

# Identify numeric and categorical features
numeric_features = [
    'Age', 'CityTier', 'NumberOfPersonVisiting', 'PreferredPropertyStar',
    'NumberOfTrips', 'NumberOfChildrenVisiting', 'MonthlyIncome',
    'PitchSatisfactionScore', 'NumberOfFollowups', 'DurationOfPitch',
    'Passport', 'OwnCar'
]

# All remaining columns are assumed to be one-hot encoded categorical columns
categorical_features = [col for col in Xtrain.columns if col not in numeric_features]

# Handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define XGBoost classifier
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=1, use_label_encoder=False, eval_metric='logloss')

# Hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Create pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Grid search with cross-validation
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1)
grid_search.fit(Xtrain, ytrain)

# Best model
best_model = grid_search.best_estimator_
print("Best Params:\n", grid_search.best_params_)

# Predict
y_pred_train = best_model.predict(Xtrain)
y_pred_test = best_model.predict(Xtest)

# Evaluation
print("\nTraining Classification Report:")
print(classification_report(ytrain, y_pred_train))

print("\nTest Classification Report:")
print(classification_report(ytest, y_pred_test))

# Save the model
model_filename = "best_tourism_model_v1.joblib"
joblib.dump(best_model, model_filename)

# Upload to Hugging Face
repo_id = "tamizh1296/tourism-package-model"
repo_type = "model"
api = HfApi(token=os.getenv("HF_TOKEN"))

# Check if repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model repo '{repo_id}' not found. Creating new repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model repo '{repo_id}' created.")

# Upload model
api.upload_file(
    path_or_fileobj=model_filename,
    path_in_repo=model_filename,
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"Model uploaded to Hugging Face repo '{repo_id}'.")
