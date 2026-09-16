# Heart-stroke-prediction
ML Project 1
# ❤️ Heart Stroke Prediction App

A simple [Streamlit](https://streamlit.io/) web app that predicts a patient's risk of heart disease using a K-Nearest Neighbors (KNN) classification model trained on clinical data.

## 🚀 Demo

Enter a patient's clinical details in the sidebar/form and click **Predict** to get an instant risk assessment: **⚠️ High Risk** or **✅ Low Risk** of heart disease.

## 🧠 How It Works

1. The user fills in numeric and categorical health parameters through the Streamlit UI.
2. The raw inputs are converted into a one-hot encoded feature row matching the format the model was trained on.
3. Missing dummy columns are filled with `0`, and columns are reordered to match `columns.pkl`.
4. The feature row is scaled using a pre-fitted `StandardScaler` (`scaler.pkl`).
5. The scaled input is passed to a pre-trained `KNeighborsClassifier` (`KNN_heart.pkl`) to generate a prediction.

## 📋 Input Features

| Feature | Description | Type |
|---|---|---|
| Age | Patient age (18–100) | Numeric |
| Resting BP | Resting blood pressure (mm Hg) | Numeric |
| Cholesterol | Serum cholesterol (mg/dl) | Numeric |
| Fasting Blood Sugar | Whether fasting blood sugar > 120 mg/dL (0/1) | Numeric |
| Max Heart Rate | Maximum heart rate achieved | Numeric |
| Oldpeak | ST depression induced by exercise | Numeric |
| Sex | M / F | Categorical |
| Chest Pain Type | ATA / NAP / ASY / TA | Categorical |
| Resting ECG | Normal / ST / LVH | Categorical |
| Exercise Angina | Y / N | Categorical |
| ST Slope | Up / Flat / Down | Categorical |

## 📂 Project Structure

```
.
├── app.py             # Streamlit application
├── KNN_heart.pkl       # Trained KNN classifier
├── scaler.pkl          # Fitted StandardScaler
├── columns.pkl          # Expected feature column order (used for one-hot alignment)
└── README.md
```

## 🛠️ Installation

1. Clone the repository

   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. Create a virtual environment (optional but recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage

Run the Streamlit app locally:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (typically `http://localhost:8501`).

## 📦 Requirements

```
streamlit
pandas
scikit-learn
joblib
```

Save the above as `requirements.txt` in the project root.

## ⚠️ Disclaimer

This tool is built for **educational/demonstration purposes only** and is **not a substitute for professional medical advice, diagnosis, or treatment**. Always consult a qualified healthcare provider for medical concerns.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
