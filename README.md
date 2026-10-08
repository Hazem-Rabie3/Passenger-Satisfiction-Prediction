# Airline Passenger Satisfaction Predictor

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-0F2C59?style=for-the-badge)

##  Overview
The **Airline Passenger Satisfaction Predictor** is a modern, interactive web application that leverages Machine Learning to predict whether an airline passenger will be satisfied with their flight experience. By analyzing various data points—ranging from passenger demographics to inflight services and airport experiences—this tool provides instant predictions to help airlines understand and improve customer satisfaction.

##  Features
*   **User-Friendly Interface:** A clean, commercial-grade UI built with Streamlit, featuring customized themes and Material Icons.
*   **Categorized Inputs:** Data entry is logically divided into intuitive tabs:
    *    Passenger & Flight Data
    *    Airport & Booking Experience
    *    Inflight Service
*   **Real-time Machine Learning:** Utilizes a pre-trained ML model (`satisfaction_model.pkl`) to instantly process user inputs and output a prediction (Satisfied vs. Neutral/Dissatisfied).
*   **Smart Defaults & Star Ratings:** Streamlined input methods to reduce user friction and save time.

##  Tech Stack
*   **Frontend:** [Streamlit](https://streamlit.io/) (with custom UI/UX configuration)
*   **Backend/Logic:** Python
*   **Data Handling:** Pandas, NumPy
*   **Machine Learning:** Scikit-Learn (Model saved via Joblib)

## Project Structure
```text
├── front.py                   # The main Streamlit application script
├── satisfaction_model.pkl     # The pre-trained machine learning model
├── .streamlit/
│   └── config.toml            # Custom UI theme configuration (Colors & Fonts)
├── requirements.txt           # List of project dependencies
└── README.md                  # Project documentation
```
##  License
This project is [MIT](https://choosealicense.com/licenses/mit/) licensed.
