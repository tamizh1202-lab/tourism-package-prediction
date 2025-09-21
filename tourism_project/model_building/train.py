import os
import pandas as pd
import joblib
import mlflow
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
import xgboost as xgb

# -----------------------
# MLflow setup
# -----------------------
mlflow.set_tracking_uri("https://f0f58b09af57.ngrok-free.app")
mlflow.set_experiment("MLOps_experiment")

# -----------------------
# Load data
# -----------------------
Xtrain = pd.read_csv("hf://datasets/tamizh1296/tourism-package-prediction/Xtrain.csv").drop(columns=['Unnamed: 0'], errors='ignore')
Xtest  = pd.read_csv("hf://datasets/tamizh1296/tourism-package-prediction/Xtest.csv").drop(columns=['Unnamed: 0'], errors='ignore')
ytrain = pd.read_csv("hf://datasets/tamizh1296/tourism-package-prediction/ytrain.csv").squeeze()
ytest  = pd.read_csv("hf://datasets/tamizh1296/tourism-package-prediction/ytest.csv").squeeze()

print("Data loaded successfully.")

numeric_features = [
    'Age','CityTier','NumberOfPersonVisiting','PreferredPropertyStar',
    'NumberOfTrips','NumberOfChildrenVisiting','MonthlyIncome',
    'PitchSatisfactionScore','NumberOfFollowups','DurationOfPitch'
]

categorical_features = [
    'Gender', 'TypeofContact', 'Passport', 'OwnCar',
    'Occupation', 'MaritalStatus', 'Designation', 'ProductPitched'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=1,
    use_label_encoder=False,
    eval_metric='logloss'
)

# -----------------------
# Full pipeline
# -----------------------
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('xgbclassifier', xgb_model)
])

# -----------------------
# Hyperparameter grid
# -----------------------
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6]
}

# -----------------------
# GridSearchCV
# -----------------------
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1)
grid_search.fit(Xtrain, ytrain)

best_model = grid_search.best_estimator_
print("Best Params:", grid_search.best_params_)

# -----------------------
# Predictions & Evaluation
# -----------------------
y_pred_train = best_model.predict(Xtrain)
y_pred_test  = best_model.predict(Xtest)

print("\nTraining Classification Report:\n", classification_report(ytrain, y_pred_train))
print("\nTest Classification Report:\n", classification_report(ytest, y_pred_test))

# -----------------------
# MLflow logging
# -----------------------
with mlflow.start_run():
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metrics({
        "train_accuracy": accuracy_score(ytrain, y_pred_train),
        "test_accuracy": accuracy_score(ytest, y_pred_test),
        "test_recall": recall_score(ytest, y_pred_test)
    })
    mlflow.sklearn.log_model(best_model, "tourism_model")

# -----------------------
# Save model locally
# -----------------------
model_filename = "best_tourism_model_v2_pipeline.joblib"
joblib.dump(best_model, model_filename)

# -----------------------
# Upload to Hugging Face
# -----------------------
api = HfApi(token=os.getenv("HF_TOKEN"))
repo_id = "tamizh1296/tourism-package-model"
repo_type = "model"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model repo '{repo_id}' created.")

api.upload_file(
    path_or_fileobj=model_filename,
    path_in_repo=model_filename,
    repo_id=repo_id,
    repo_type=repo_type
)
print(f"Model uploaded to Hugging Face repo '{repo_id}'.")
