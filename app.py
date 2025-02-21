import logging
import warnings
from pathlib import Path
from flask import Flask, request, render_template
from markupsafe import Markup 
import pandas as pd
import numpy as np
import pickle
import json
import requests
from config import *
from fertilizer import fertilizer_dic

DISTRICT_TO_STATE = {
    'adilabad': 'telangana',
    'bhadradri kothagudem': 'telangana',
    'hanamkonda': 'telangana',
    'hyderabad': 'telangana',
    'jagtial': 'telangana',
    'janagama': 'telangana',
    'jayashankar bhupalpally': 'telangana',
    'jogulamba gadwal': 'telangana',
    'kamareddy': 'telangana',
    'karimnagar': 'telangana',
    'khammam': 'telangana',
    'komaram bheem asifabad': 'telangana',
    'mahabubabad': 'telangana',
    'mahabubnagar': 'telangana',
    'mancherial': 'telangana',
    'medak': 'telangana',
    'medchal–malkajgiri': 'telangana',
    'mulugu': 'telangana',
    'nagarkurnool': 'telangana',
    'nalgonda': 'telangana',
    'narayanpet': 'telangana',
    'nirmal': 'telangana',
    'nizamabad': 'telangana',
    'peddapalli': 'telangana',
    'rajanna sircilla': 'telangana',
    'rangareddy': 'telangana',
    'sangareddy': 'telangana',
    'siddipet': 'telangana',
    'suryapet': 'telangana',
    'vikarabad': 'telangana',
    'wanaparthy': 'telangana',
    'warangal': 'telangana',
    'yadadri bhuvanagiri': 'telangana',
}


# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

try:
    # Validate all required files exist
    validate_paths()
    
    # Load models and data
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(LABELS_PATH, 'rb') as f:
        labels = pickle.load(f)
    with open(STATES_SEASON_PATH, 'r') as f:
        common_label = json.load(f)
        
    logger.info("Successfully loaded all models and data files")
except Exception as e:
    logger.error(f"Failed to initialize application: {e}")
    raise

# Route handlers
@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/choose')
def choose():
    return render_template('Choose.html')

@app.route('/details')
def details():
    return render_template('Details.html')

@app.route('/fertilizer_details')
def fertilizer_details():
    return render_template('Fertilizer_details.html')

@app.route('/predict', methods=['POST'])
def prediction():
    try:
        # Input validation
        required_fields = ['temp', 'humidity', 'ph', 'rainfall', 'season', 'state', 
                         'nitrogen', 'phosphorous', 'potassium']
        
        for field in required_fields:
            if field not in request.form:
                raise ValueError(f"Missing required field: {field}")

        state = request.form['state'].lower()
        season = request.form['season'].lower()

        # Validate state
        if state not in common_label:
            raise ValueError(f"Invalid state: {state}. Please select from the dropdown list.")

        # Validate season
        if season not in common_label:
            raise ValueError(f"Invalid season: {season}. Please select from the dropdown list.")

        # Parse and validate numeric inputs
        query = np.array([
            float(request.form['temp']),
            float(request.form['humidity']), 
            float(request.form['ph']),
            float(request.form['rainfall']),
            common_label[season],
            common_label[state],
            float(request.form['nitrogen']),
            float(request.form['phosphorous']),
            float(request.form['potassium'])
        ]).reshape(1, -1)

        # Get predictions
        result = model.predict_proba(query)[0]
        top_gainers = labels.inverse_transform(np.argsort(-result)[:5])
        top_loosers = labels.inverse_transform(np.argsort(result)[:5])

        return render_template('Crop_data.html', 
                             top_gainers=top_gainers,
                             top_loosers=top_loosers)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template('error.html', error=str(e))

@app.route('/fertilizers', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        crop_name = str(request.form['crop'])
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['potassium'])
        # ph = float(request.form['ph'])

        df = pd.read_csv(r"E:\Projects\Smart_Harvest\data\fertilizer.csv")

        nr = df[df['Crop'] == crop_name]['N'].iloc[0]
        pr = df[df['Crop'] == crop_name]['P'].iloc[0]
        kr = df[df['Crop'] == crop_name]['K'].iloc[0]

        n = nr - N
        p = pr - P
        k = kr - K
        temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
        max_value = temp[max(temp.keys())]
        if max_value == "N":
            if n < 0:
                key = 'NHigh'
            else:
                key = "Nlow"
        elif max_value == "P":
            if p < 0:
                key = 'PHigh'
            else:
                key = "Plow"
        else:
            if k < 0:
                key = 'KHigh'
            else:
                key = "Klow"

        response = Markup(str(fertilizer_dic[key]))
        return render_template('Fertilizer_result.html', response = response)
    return render_template('Fertilizer_details.html')

# Weather
@app.route('/weather', methods=['POST'])
def weather():
    try:
        city = request.form['city'].lower().strip()  # Convert to lowercase and remove whitespace
        state = request.form.get('state', '').lower().strip()

        # If city is a district, use its state mapping
        if city in DISTRICT_TO_STATE:
            state = DISTRICT_TO_STATE[city]
        elif not state:  # If no state is provided and city isn't a district
            state = city  # Fallback to using city as state

        # Validate state
        if state not in common_label:
            raise ValueError(f"Invalid location. Please select a valid state or district.")
        
        # Use OpenMeteo free API which doesn't require key
        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": 20.5937,  # Default to India coordinates
                "longitude": 78.9629,
                "current_weather": True,
                "hourly": "temperature_2m,relativehumidity_2m"
            },
            timeout=5
        )
        weather_response.raise_for_status()
        
        weather_data = weather_response.json()
        
        return render_template(
            'Details.html',
            temperature=str(weather_data['current_weather']['temperature']),
            humidity=str(weather_data['hourly']['relativehumidity_2m'][0]),
            state=state  # Use mapped state instead of city directly
        )

    except Exception as e:
        logger.error(f"Weather API error: {e}")
        return render_template('error.html', error="Unable to fetch weather data")

@app.route('/crop_info')
def crop_info():
    return render_template('Crop_info.html')

@app.route("/goback")
def goback():
    return render_template('Choose.html')


if __name__ == '__main__':
    app.run(debug=True)