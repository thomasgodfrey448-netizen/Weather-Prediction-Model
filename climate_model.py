from flask import Flask, render_template, request
import joblib
import pandas as pd

# 1. Create Flask app
weather = Flask(__name__)

# 2. Load the trained model
d_model = joblib.load("Weather_prediction_model.pkl")

# 3. Month mapping for user input
month_map = {
    "january": 1, "february": 2, "march": 3,
    "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9,
    "october": 10, "november": 11, "december": 12
}

# 4. Home route to render HTML form
@weather.route("/")
def home():
    return render_template("index.html")

# 5. Prediction route to handle form submission and return prediction
@weather.route("/predict", methods=["POST"])
def predict():

    # Get form inputs from HTML
    month_name = request.form["month"].strip().lower()
    precipitation = float(request.form["precipitation"])
    temp_max = float(request.form["temp_max"])
    temp_min = float(request.form["temp_min"])
    wind = float(request.form["wind"])

    # Convert month name → number
    if month_name not in month_map:
        return "Invalid month name"

    month = month_map[month_name]

    temp_diff = temp_max - temp_min

    # Create DataFrame for model
    new_data = pd.DataFrame([{
        "precipitation": precipitation,
        "temp_max": temp_max,
        "temp_min": temp_min,
        "wind": wind,
        "month": month
    }])

    # Prediction
    prediction = d_model.predict(new_data)

    # Return result to webpage
    return render_template(
        "index.html",
        prediction=prediction,
        month_name=month_name,
        precipitation=precipitation,
        temp_diff=temp_diff,
        temp_max=temp_max,
        temp_min=temp_min,
        wind=wind
    )

# 6. Run server
if __name__ == "__main__":
    weather.run(debug=True)