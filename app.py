import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os

model_path = os.path.join(os.path.dirname(__file__), "airline_recommendation_model.pkl")

model = pickle.load(open(model_path, 'rb'))

st.set_page_config(page_title="Airline Recommendation Dashboard", layout="wide")

# ---------- CSS ----------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0b0f1a;
    color: white;
}
.hero {
    background-image: url('https://images.unsplash.com/photo-1542296332-2e4473faf563');
    background-size: cover;
    background-position: center;
    padding: 80px;
    border-radius: 20px;
    color: white;
}
.hero-overlay {
    background: rgba(0,0,0,0.6);
    padding: 40px;
    border-radius: 20px;
}
.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 25px;
    margin-top: -50px;
}
.stButton>button {
    background-color: #e63946;
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ---------- Hero Section ----------
st.markdown("""
<div class="hero">
    <div class="hero-overlay">
        <h1>✈️ Airline Passenger Recommendation Dashboard</h1>
        <h4>Analyze customer experience and predict recommendations</h4>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Airline List ----------
airlines = [
'Turkish Airlines','Qatar Airways','Emirates','Lufthansa',
'KLM Royal Dutch Airlines','Virgin America','American Airlines',
'Delta Air Lines','Southwest Airlines','United Airlines',
'Jetblue Airways','Aegean Airlines','Aeroflot Russian Airlines',
'Aeromexico','Air Canada','Air New Zealand','Alitalia',
'AirAsia','Asiana Airlines','Avianca','Austrian Airlines',
'British Airways','Brussels Airlines','China Eastern Airlines',
'China Southern Airlines','Copa Airlines','Ethiopian Airlines',
'Egyptair','Finnair','Iberia','ANA All Nippon Airways',
'easyJet','Korean Air','LATAM Airlines','LOT Polish Airlines',
'Qantas Airways','Air France','Etihad Airways',
'Pegasus Airlines','Royal Jordanian Airlines','Ryanair',
'South African Airways','Saudi Arabian Airlines','TAP Portugal',
'Eurowings','EVA Air','Royal Air Maroc','Singapore Airlines',
'SAS Scandinavian','Swiss Intl Air Lines','Thai Airways',
'Air India','Air Europa','Air Canada rouge','airBaltic',
'Air China','Cathay Pacific Airways','Wizz Air',
'Spirit Airlines','TAROM Romanian','Vueling Airlines',
'Sunwing Airlines','QantasLink','Bangkok Airways','flydubai',
'Garuda Indonesia','Germanwings','Frontier Airlines',
'Icelandair','IndiGo','Aer Lingus','Adria Airways',
'Air Arabia','Alaska Airlines','Tunisair','Norwegian',
'Thai Smile Airways','Gulf Air','Kuwait Airways','WOW air',
'Ukraine International'
]

# ---------- Glass Card ----------
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("✈️ Passenger Experience Input")

# Select Airline
airline = st.selectbox("Select Airline", airlines)

# Input ratings
col1, col2 = st.columns(2)
with col1:
    cabin_service = st.slider("Cabin Service", 1, 5, 3)
    seat_comfort = st.slider("Seat Comfort", 1, 5, 3)
    food_bev = st.slider("Food & Beverage", 1, 5, 3)
with col2:
    entertainment = st.slider("Entertainment", 1, 5, 3)
    ground_service = st.slider("Ground Service", 1, 5, 3)
    value_for_money = st.slider("Value for Money", 1, 5, 3)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- File to store all passengers ----------
file_name = "passenger_data.csv"
if not os.path.exists(file_name):
    df_init = pd.DataFrame(columns=["Airline","Cabin","Seat","Food","Entertainment","Ground","Value","Prediction","Probability"])
    df_init.to_csv(file_name,index=False)

# ---------- Predict ----------
if st.button("🔍 Predict Recommendation"):
    input_data = np.array([[cabin_service, seat_comfort, food_bev,
                            entertainment, ground_service, value_for_money]])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Show result for this passenger
    if prediction == 1:
        st.success(f"✅ Passenger WILL recommend {airline}")
    else:
        st.error(f"❌ Passenger will NOT recommend {airline}")

    st.info(f"📊 Recommendation Probability: {round(probability*100,2)}%")

    # ---------- Save to CSV ----------
    new_data = pd.DataFrame({
        "Airline":[airline],
        "Cabin":[cabin_service],
        "Seat":[seat_comfort],
        "Food":[food_bev],
        "Entertainment":[entertainment],
        "Ground":[ground_service],
        "Value":[value_for_money],
        "Prediction":[prediction],
        "Probability":[probability]
    })
    new_data.to_csv(file_name, mode='a', header=False, index=False)

    # ---------- Load all data ----------
    data = pd.read_csv(file_name)

    # Filter for selected airline
    airline_data = data[data["Airline"]==airline]

    st.subheader(f"📈 {airline} - Dashboard Summary")

    if len(airline_data) > 0:
        # Recommendation Rate
        rec_rate = airline_data["Prediction"].mean()*100
        st.metric("Overall Recommendation Rate", f"{rec_rate:.2f}%")

        # Average ratings
        avg_ratings = airline_data[["Cabin","Seat","Food","Entertainment","Ground","Value"]].mean()
        st.write("### Average Ratings")
        st.bar_chart(avg_ratings)

        # Weakest area(s) - can be multiple if same rating
        min_value = avg_ratings.min()
        weakest_areas = avg_ratings[avg_ratings == min_value].index.tolist()

        weakest_text = ", ".join([f"{area} ({avg_ratings[area]:.2f})" for area in weakest_areas])
        st.warning(f"⚠️ Weakest Area(s): {weakest_text}")
