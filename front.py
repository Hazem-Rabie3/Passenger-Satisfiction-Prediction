import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ─── Page Configuration (Must be the first Streamlit command) ───
st.set_page_config(page_title="Airline Satisfaction Predictor", page_icon="✈️", layout="wide")

# ─── Custom CSS for UI/UX Enhancement ───
st.markdown("""
    <style>
    /* Clean, modern font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Calming background gradient inspired by the sky */
    .stApp {
        background: linear-gradient(180deg, #f0f8ff 0%, #e0f2fe 100%);
    }
    
    /* Primary Call-to-Action button styling */
    div.stButton > button {
        background-color: #0284c7 !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 0.5rem 1rem !important;
        font-size: 20px !important;
        font-weight: bold !important;
        width: 100%;
        border: none !important;
        transition: all 0.3s ease-in-out;
    }
    div.stButton > button:hover {
        background-color: #0369a1 !important;
        transform: scale(1.02);
    }
    
    /* Header coloring */
    h1, h2, h3 {
        color: #0c4a6e !important;
    }
    
    /* Tab styling */
    button[data-baseweb="tab"] {
        font-size: 25px !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ─── Smart Model Loading (Caches model to speed up the app) ───
@st.cache_resource
def load_model():
    return joblib.load('satisfaction_model.pkl')

model = load_model()

# ─── App Header ───
st.title("✈️ Airline Passenger Satisfaction Predictor")
st.markdown("### 🛫 Enter passenger flight details to accurately predict their satisfaction level")
st.write("---")

# ─── Helper function for Star Ratings (Replaces basic sliders) ───
def star_rating(label, default_val=3):
    return st.selectbox(
        label,
        options=[5, 4, 3, 2, 1, 0],
        index=5 - default_val, # Sets default to 3
        format_func=lambda x: f"{x} - " + ("⭐" * x) if x > 0 else "0 - N/A / Very Poor"
    )

# ─── Organize inputs into Tabs for a cleaner layout ───
tab1, tab2, tab3 = st.tabs(["🧑‍💼 Passenger & Flight Data", "🏢 Airport & Booking Experience", "✈️ Inflight Service"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
        age = st.number_input("Age", min_value=7, max_value=100, value=30, step=1)
        travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
        
    with col2:
        travel_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
        flight_distance = st.number_input("Flight Distance (km)", 100, 10000, 1000, step=50)
        departure_delay = st.number_input("Departure Delay (Minutes)", 0, 2000, 0, step=5)
        # Defaults to departure_delay to save user time, but allows independent input
        arrival_delay = st.number_input("Arrival Delay (Minutes)", 0, 2000, departure_delay, step=5)

with tab2:
    st.info("💡 Rate the following services from 1 to 5 stars:")
    c1, c2 = st.columns(2)
    with c1:
        ease_online_booking = star_rating("Ease of Online booking")
        online_boarding = star_rating("Online boarding")
        departure_arrival_time = star_rating("Departure/Arrival time convenient")
    with c2:
        checkin_service = star_rating("Check-in service")
        gate_location = star_rating("Gate location")
        baggage_handling = star_rating("Baggage handling")

with tab3:
    st.info("💡 Rate the inflight experience:")
    c3, c4 = st.columns(2)
    with c3:
        seat_comfort = star_rating("Seat comfort")
        leg_room = star_rating("Leg room service")
        food_drink = star_rating("Food and drink")
        cleanliness = star_rating("Cleanliness")
    with c4:
        inflight_wifi = star_rating("Inflight wifi service")
        entertainment = star_rating("Inflight entertainment")
        onboard_service = star_rating("On-board service")
        inflight_service = star_rating("Inflight service")

st.write("---")

# ─── Prediction Section ───
# Center the button for better aesthetics
_, center_col, _ = st.columns([1, 2, 1])

with center_col:
    predict_btn = st.button("🔮 Predict Satisfaction Now")

if predict_btn:
    with st.spinner("Analyzing data... ⏳"):
        input_data = pd.DataFrame([{
            'Gender': gender,
            'Customer Type': customer_type,
            'Age': age,
            'Type of Travel': travel_type,
            'Class': travel_class,
            'Flight Distance': flight_distance,
            'Inflight wifi service': inflight_wifi,
            'Departure/Arrival time convenient': departure_arrival_time,
            'Ease of Online booking': ease_online_booking,
            'Gate location': gate_location,
            'Food and drink': food_drink,
            'Online boarding': online_boarding,
            'Seat comfort': seat_comfort,
            'Inflight entertainment': entertainment,
            'On-board service': onboard_service,
            'Leg room service': leg_room,
            'Baggage handling': baggage_handling,
            'Checkin service': checkin_service,
            'Inflight service': inflight_service,
            'Cleanliness': cleanliness,
            'Departure Delay in Minutes': departure_delay,
            'Arrival Delay in Minutes': arrival_delay
        }])

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        
        # Adjust based on your model's specific class mapping
        satisfaction_proba = proba[1] if prediction == 1 else proba[0]

        st.write("") # Add spacing
        
        # Display results with visual feedback
        if prediction == 1:
            st.success(f"###  Result: The passenger is likely to be **Satisfied**!")
            st.balloons() # Visual celebration for positive result
        else:
            st.error(f"###  Result: The passenger is likely to be **Neutral or Dissatisfied**.")
