import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ─── Page Configuration ───
st.set_page_config(
    page_title="Airline Satisfaction Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS for UI/UX Enhancement ───
st.markdown("""
    <style>
    /* Global Styling */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Headers */
    h1 {
        color: #025091 !important;
        font-weight: 800 !important;
        text-align: center;
        padding-bottom: 20px;
    }
    h2, h3 {
        color: #025091 !important;
        font-weight: 600 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 8px 8px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #0F172A !important;
        font-weight: 600 !important;
        font-size: 18px !important;
        box-shadow: 0px -2px 5px rgba(0,0,0,0.05);
        border: 1px solid #E2E8F0;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #F8FAFC !important;
        border-top: 4px solid #025091 !important;
        color: #025091 !important;
    }

    /* Result card */
    .result-card {
        background-color: #ffffff;
        padding: 32px;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 24px rgba(2, 80, 145, 0.08);
        text-align: center;
        max-width: 640px;
        margin: 0 auto;
    }
    .result-satisfied {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border-color: #34D399;
    }
    .result-dissatisfied {
        background: linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 100%);
        border-color: #F87171;
    }
    .result-label {
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 8px 0;
    }
    .result-label.satisfied { color: #065F46; }
    .result-label.dissatisfied { color: #991B1B; }
    .result-sub {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 20px;
    }
    .confidence-label {
        font-size: 0.9rem;
        color: #64748b;
        margin-top: 6px;
    }
    
    /* CTA Button section - no card */
    .cta-section {
        padding: 24px 0;
    }

    /* Button styling */
    div.stButton > button {
        background-color: #025091 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.8rem 2rem !important;
        font-size: 20px !important;
        font-weight: bold !important;
        width: 100%;
        border: none !important;
        box-shadow: 0 4px 6px -1px rgba(2, 80, 145, 0.4);
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background-color: #013b6e !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px -1px rgba(2, 80, 145, 0.6);
    }
    
    div.stButton > button:focus {
        outline: 3px solid #60A5FA !important;
    }

    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #025091;
        font-weight: 700;
    }
    
    /* Input widgets text contrast */
    .stSelectbox label, .stNumberInput label, .stRadio label, .stSlider label {
        color: #0F172A !important;
        font-weight: 600 !important;
    }
    
    /* Radio options */
    div[role="radiogroup"] label {
        color: #1E293B !important;
        font-weight: 500 !important;
    }
    
    /* Fix horizontal radio container for better pill-like appearance */
    div[role="radiogroup"] {
        gap: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ─── Model Loading ───
@st.cache_resource
def load_model():
    try:
        return joblib.load('satisfaction_model.pkl')
    except Exception as e:
        return None

model = load_model()

# ─── App Header ───
st.markdown("<h1>✈️ Airline Passenger Satisfaction Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #475569; font-size: 1.2rem; margin-bottom: 30px;'>Enhance customer experience by predicting passenger satisfaction based on their journey details.</p>", unsafe_allow_html=True)

# ─── Helpers ───
def star_rating(label, key):
    # Default to 3 if not set (st.feedback returns 0-4 or None)
    val = st.feedback("stars", key=key)
    score = (val + 1) if val is not None else 3
    st.markdown(f"<div style='margin-top: -15px; margin-bottom: 15px; font-weight: 600; color: #0F172A;'>{label} <span style='color: #64748b; font-weight: 400;'>({score}/5)</span></div>", unsafe_allow_html=True)
    return score

# ─── Main Content ───
if model is None:
    st.error("⚠️ Model file 'satisfaction_model.pkl' not found. Please ensure the model is trained and saved in the same directory.")
else:
    # Use tabs for a clean UI
    tab1, tab2, tab3 = st.tabs(["📋 Passenger Details", "🛫 Airport Experience", "☁️ Inflight Services"])
    
    with tab1:
        st.markdown("### 🧑‍💼 Passenger & Flight Information")
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
            customer_type = st.radio("Customer Loyalty", ["Loyal Customer", "disloyal Customer"], horizontal=True)
            age = st.slider("Age", min_value=7, max_value=100, value=30, step=1)
            travel_type = st.radio("Purpose of Travel", ["Business travel", "Personal Travel"], horizontal=True)
            
        with col2:
            travel_class = st.radio("Cabin Class", ["Business", "Eco", "Eco Plus"], horizontal=True)
            flight_distance = st.number_input("Flight Distance (km)", min_value=50, max_value=15000, value=1000, step=50)
            
            c_del1, c_del2 = st.columns(2)
            with c_del1:
                departure_delay = st.number_input("Departure Delay (min)", 0, 2000, 0, step=5)
            with c_del2:
                arrival_delay = st.number_input("Arrival Delay (min)", 0, 2000, departure_delay, step=5)

    with tab2:
        st.markdown("### 🏢 Airport & Booking Services")
        st.markdown("<p style='color: #475569;'>Rate the services provided before the flight</p>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            ease_online_booking = star_rating("Ease of Online Booking", "ease_online_booking")
            online_boarding = star_rating("Online Boarding", "online_boarding")
        with c2:
            departure_arrival_time = star_rating("Convenient Departure/Arrival", "departure_arrival_time")
            checkin_service = star_rating("Check-in Service", "checkin_service")
        with c3:
            gate_location = star_rating("Gate Location", "gate_location")
            baggage_handling = star_rating("Baggage Handling", "baggage_handling")

    with tab3:
        st.markdown("### ✈️ Inflight Experience")
        st.markdown("<p style='color: #475569;'>Rate the services provided during the flight</p>", unsafe_allow_html=True)
        
        c4, c5, c6 = st.columns(3)
        with c4:
            seat_comfort = star_rating("Seat Comfort", "seat_comfort")
            leg_room = star_rating("Leg Room", "leg_room")
            cleanliness = star_rating("Cleanliness", "cleanliness")
        with c5:
            food_drink = star_rating("Food & Drink", "food_drink")
            inflight_wifi = star_rating("Inflight Wi-Fi", "inflight_wifi")
            entertainment = star_rating("Entertainment", "entertainment")
        with c6:
            onboard_service = star_rating("On-board Service", "onboard_service")
            inflight_service = star_rating("Overall Inflight Service", "inflight_service")

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # ─── Separator ───
    st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 0 0 24px 0;'>", unsafe_allow_html=True)

    # ─── Prediction Button — centered, no flanking columns ───
    st.markdown("<div style='max-width: 400px; margin: 0 auto;'>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Analyze Passenger Satisfaction", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ─── Prediction Logic & Output ───
    if predict_btn:
        input_dict = {
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
        }
        
        input_data = pd.DataFrame([input_dict])
        
        with st.spinner("Analyzing profile..."):
            try:
                prediction = model.predict(input_data)[0]
                try:
                    proba = model.predict_proba(input_data)[0]
                    confidence = proba[1] if prediction == 1 else proba[0]
                except:
                    confidence = None

                st.markdown("<br>", unsafe_allow_html=True)
                
                satisfied = (prediction == 1 or str(prediction).lower() == 'satisfied')
                
                if satisfied:
                    card_class = "result-card result-satisfied"
                    emoji = "✨"
                    label_class = "result-label satisfied"
                    result_text = "Likely Satisfied"
                    sub_text = "The passenger profile indicates a positive experience."
                else:
                    card_class = "result-card result-dissatisfied"
                    emoji = "😞"
                    label_class = "result-label dissatisfied"
                    result_text = "Neutral or Dissatisfied"
                    sub_text = "This profile suggests dissatisfaction. Consider reaching out or offering a service recovery perk."
                
                # ── Card header (no nested f-string for conf_html) ──
                st.markdown(
                    f'<div class="{card_class}">'
                    f'<p style="font-size:2.4rem;margin:0 0 4px 0;">{emoji}</p>'
                    f'<p class="{label_class}">{result_text}</p>'
                    f'<p class="result-sub">{sub_text}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # ── Confidence bar rendered separately ──
                if confidence is not None:
                    pct = int(confidence * 100)
                    bar_color = "#34D399" if satisfied else "#F87171"
                    st.markdown(
                        f'<div style="max-width:640px;margin:12px auto 0 auto;">'
                        f'<div style="background:#E2E8F0;border-radius:999px;height:10px;overflow:hidden;">'
                        f'<div style="width:{pct}%;background:{bar_color};height:100%;border-radius:999px;"></div>'
                        f'</div>'
                        f'<p style="text-align:center;font-size:0.9rem;color:#64748b;margin-top:6px;">'
                        f'Model Confidence: <strong>{pct}%</strong></p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                if satisfied:
                    st.balloons()

            except Exception as e:
                st.markdown(
                    f'<div class="result-card" style="border-color:#F87171;background:#FFF1F2;">'
                    f'<p style="color:#991B1B;font-weight:700;font-size:1.1rem;">⚠️ Prediction Error</p>'
                    f'<p style="color:#475569;">{e}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

