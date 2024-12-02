import pandas as pd
from datetime import datetime
import streamlit as st
import pickle

with open('lr.pkl', 'rb') as file:
    maintenance_model = pickle.load(file)
with open('ln.pkl', 'rb') as file:
    mileage_model = pickle.load(file)

st.title("Vehicle Maintenance and Mileage Prediction App")

Vehicle_Model = st.sidebar.selectbox("Vehicle Model", ('Bus', 'Car', 'Motorcycle', 'SUV', 'Truck', 'Van'))
Maintenance_History = st.sidebar.selectbox("Maintenance History", ('Good', 'Average', 'Poor'))
Reported_Issues = st.sidebar.slider("Reported Issues", min_value=0, max_value=10)
Vehicle_Age = st.sidebar.slider("Vehicle Age (years)", min_value=0, max_value=30)
Fuel_Type = st.sidebar.selectbox("Fuel Type", ('Diesel', 'Petrol', 'Electric'))
Transmission_Type = st.sidebar.selectbox("Transmission Type", ('Manual', 'Automatic'))
Engine_Size = st.sidebar.number_input("Engine Size (cc)", min_value=25)
Odometer_Reading = st.sidebar.number_input("Odometer Reading (km)", min_value=0)
Last_Service_Date = st.sidebar.date_input("Last Service Date")
Warranty_Expiry_Date = st.sidebar.date_input("Warranty Expiry Date")
Owner_Type = st.sidebar.selectbox("Owner Type", ('First', 'Second', 'Third'))
Service_History = st.sidebar.slider("No. of Services Done", min_value=0, max_value=30)
Accident_History = st.sidebar.slider("Accident History", min_value=0, max_value=5)
Tire_Condition = st.sidebar.selectbox("Tire Condition", ('New', 'Worn Out', 'Good'))
Brake_Condition = st.sidebar.selectbox("Brake Condition", ('New', 'Worn Out', 'Good'))
Battery_Status = st.sidebar.selectbox("Battery Status", ('New', 'Good', 'Weak'))

vehicle_model_features = {f'Vehicle_Model_{model}': 0 for model in ['Bus', 'Car', 'Motorcycle', 'SUV', 'Truck', 'Van']}
vehicle_model_features[f'Vehicle_Model_{Vehicle_Model}'] = 1

maintenance_history_features = {f'Maintenance_History_{status}': 0 for status in ['Good', 'Average', 'Poor']}
maintenance_history_features[f'Maintenance_History_{Maintenance_History}'] = 1

today = datetime.today().date()
last_service_date = pd.to_datetime(Last_Service_Date).date()
days_without_service = (today - last_service_date).days

warranty_expiry_date = pd.to_datetime(Warranty_Expiry_Date).date()
warranty_left = (warranty_expiry_date - today).days

fuel_type_features = {f'Fuel_Type_{fuel}': int(Fuel_Type == fuel) for fuel in ['Diesel', 'Petrol', 'Electric']}
transmission_type_features = {f'Transmission_Type_{trans}': int(Transmission_Type == trans) for trans in ['Manual', 'Automatic']}
owner_type_features = {f'Owner_Type_{owner}': int(Owner_Type == owner) for owner in ['First', 'Second', 'Third']}
tire_condition_features = {f'Tire_Condition_{condition}': int(Tire_Condition == condition) for condition in ['Good', 'New', 'Worn Out']}
brake_condition_features = {f'Brake_Condition_{condition}': int(Brake_Condition == condition) for condition in ['Good', 'New', 'Worn Out']}
battery_status_features = {f'Battery_Status_{status}': int(Battery_Status == status) for status in ['Good', 'New', 'Weak']}

input_data = {
    'Reported_Issues': Reported_Issues,
    'Vehicle_Age': Vehicle_Age,
    'Engine_Size': Engine_Size,
    'Odometer_Reading': Odometer_Reading,
    'Service_History': Service_History,
    'Accident_History': Accident_History,
    'days_without_service': days_without_service,
    'warranty_left': warranty_left,
    **vehicle_model_features,
    **maintenance_history_features,
    **fuel_type_features,
    **transmission_type_features,
    **owner_type_features,
    **tire_condition_features,
    **brake_condition_features,
    **battery_status_features
}

expected_features = maintenance_model.feature_names_in_
input_df = pd.DataFrame([input_data], columns=expected_features)

st.subheader("User Input")
st.write(input_df)

if st.sidebar.button("Predict Maintenance"):
    maintenance_prediction = maintenance_model.predict(input_df)
    need_maintenance = int(maintenance_prediction[0])
    result = "Vehicle needs maintenance" if need_maintenance else "Vehicle doesn't need maintenance"
    st.subheader("Maintenance Prediction")
    st.write(result)
    maintenance_proba = maintenance_model.predict_proba(input_df)
    st.subheader("Prediction Probability")
    st.write(f"Probability that your vehicle requires maintenance: {maintenance_proba[0][1] * 100:.2f}%")

    input_data['Need_Maintenance'] = need_maintenance
    input_df_with_maintenance = pd.DataFrame([input_data], columns=mileage_model.feature_names_in_)

    mileage_prediction = mileage_model.predict(input_df_with_maintenance)
    st.subheader("Mileage Prediction")
    st.write(f"Predicted mileage: {mileage_prediction[0]:.2f} km/l")
