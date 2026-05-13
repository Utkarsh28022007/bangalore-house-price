# 🏠 Bangalore House Price Predictor

A machine learning web app that predicts house prices in Bangalore.
Built with Python, scikit-learn, and Streamlit.

- **R² Score:** 0.75
- **Dataset:** Bengaluru House Price Data (Kaggle)
- **Model:** Random Forest Regressor

## 📁 Project Structure

```
bangalore-house-price/
│
├── model/
│   ├── house_price_model.pkl   ← trained ML model
│   └── columns.pkl             ← feature column names
│
├── notebook/
│   └── Untitled7.ipynb         ← training notebook
│
├── data/
│   └── Bengaluru_House_Data.csv
│
├── app.py                      ← Streamlit web app
├── requirements.txt            ← Python dependencies
└── README.md
```

## 🚀 How to Run

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Run the app:
```
streamlit run app.py
```

3. Open browser at: http://localhost:8501
