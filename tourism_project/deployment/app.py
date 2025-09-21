%%writefile tourism_project/deployment/app.py
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model from Hugging Face
model_path = hf_hub_download(
    repo_id="tamizh1296/tourism-package-model",
    filename="best_tourism_model_v1.joblib"
)
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Package Purchase Prediction")
st.write("Predict whether a customer is likely to purchase a tourism package.")

# --- User inputs ---
age = st.number_input("Age", 18, 100, 30)
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Large Business", "Small Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_person_visiting = st.number_input("Number of Persons Visiting", 1, 10, 2)
preferred_property_star = st.number_input("Preferred Property Star", 1, 5, 3)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced","Unmarried"])
num_trips = st.number_input("Number of Trips per Year", 0, 50, 3)
passport = st.selectbox("Has Passport?", ["No", "Yes"])
own_car = st.selectbox("Owns Car?", ["No", "Yes"])
num_children_visiting = st.number_input("Number of Children (below 5) Visiting", 0, 5, 0)
designation = st.selectbox("Designation", ["Manager", "Executive", "AVP", "VP", "Senior Manager"])
monthly_income = st.number_input("Monthly Income", 5000, 1000000, 50000)
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", 0, 10, 7)
product_pitched = st.selectbox("Product Pitched", ["Basic", "King", "Standard", "Deluxe", "Super Deluxe"])
num_followups = st.number_input("Number of Follow-ups", 0, 20, 2)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", 1, 180, 15)

# --- Assemble raw input into DataFrame ---
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_person_visiting,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": num_followups,
    "DurationOfPitch": duration_of_pitch
}])

# --- Prediction ---
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Will Purchase Package" if prediction == 1 else "Will Not Purchase Package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
