import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model from Hugging Face
model_path = hf_hub_download(
    repo_id="tamizh1296/tourism-package-model", 
    filename="best_tourism_model_v1.joblib",
    repo_type="model"
)
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer is likely to purchase a tourism package.
Please enter the customer details below to get a prediction.
""")

# --- User inputs ---
age = st.number_input("Age", min_value=18, max_value=100, value=30)
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Large Business", "Small Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
preferred_property_star = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced","Unmarried"])
num_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=50, value=3)
passport = st.selectbox("Has Passport?", ["No", "Yes"])
own_car = st.selectbox("Owns Car?", ["No", "Yes"])
num_children_visiting = st.number_input("Number of Children (below 5) Visiting", min_value=0, max_value=5, value=0)
designation = st.selectbox("Designation", ["Manager", "Executive", "AVP", "VP", "Senior Manager"])
monthly_income = st.number_input("Monthly Income", min_value=5000, max_value=1000000, value=50000)
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", min_value=0, max_value=10, value=7)
product_pitched = st.selectbox("Product Pitched", ["Basic", "King", "Standard", "Deluxe", "Super Deluxe"])
num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=2)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=180, value=15)

# --- Assemble input ---
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': 0 if typeof_contact=="Company Invited" else 1,
    'CityTier': city_tier,
    'Gender': 0 if gender=="Male" else 1,
    'NumberOfPersonVisiting': num_person_visiting,
    'PreferredPropertyStar': preferred_property_star,
    'NumberOfTrips': num_trips,
    'Passport': 0 if passport=="No" else 1,
    'OwnCar': 0 if own_car=="No" else 1,
    'NumberOfChildrenVisiting': num_children_visiting,
    'MonthlyIncome': monthly_income,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'NumberOfFollowups': num_followups,
    'DurationOfPitch': duration_of_pitch,
    # For one-hot encoded categorical columns, set 1 for selected category, 0 for others
    f'Occupation_{occupation}': 1,
    f'MaritalStatus_{marital_status}': 1,
    f'Designation_{designation}': 1,
    f'ProductPitched_{product_pitched}': 1
}])

# --- Prediction ---
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Will Purchase Package" if prediction==1 else "Will Not Purchase Package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
