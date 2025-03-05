from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

app = FastAPI()

# Define input schema
class PredictionRequest(BaseModel):
    openai_api_key: str

# Create LSTM model dynamically
def create_lstm_model():
    model = Sequential([
        LSTM(64, return_sequences=True, activation="relu", input_shape=(5, 3)),
        LSTM(32, return_sequences=False, activation="relu"),
        Dense(16, activation="relu"),
        Dense(3)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0), loss="mse")
    return model

model = create_lstm_model()
scalers = []

# Fetch Windborne data and fill missing values
def fetch_windborne_data(hours=24):
    all_data = {}
    for h in range(hours):
        url = f"https://a.windbornesystems.com/treasure/{h:02d}.json"
        response = requests.get(url)
        if response.status_code == 200:
            try:
                data = response.json()
                if isinstance(data, list):
                    all_data[h] = np.array(data)
            except:
                continue
    
    if not all_data:
        return None
    
    num_balloons = len(next(iter(all_data.values())))
    complete_data = np.full((hours, num_balloons, 3), np.nan)
    
    for h, data in all_data.items():
        complete_data[h] = data
    
    # Fill missing values with column mean
    for b in range(num_balloons):
        for d in range(3):
            valid_mask = ~np.isnan(complete_data[:, b, d])
            valid_times = np.where(valid_mask)[0]
            if len(valid_times) > 0:
                mean_value = np.nanmean(complete_data[:, b, d])
                complete_data[:, b, d] = np.where(np.isnan(complete_data[:, b, d]), mean_value, complete_data[:, b, d])
    
    return complete_data

# Prepare data for LSTM
def process_sequences(data, time_steps=5):
    num_balloons = data.shape[1]
    X, y = [], []
    scalers = [MinMaxScaler() for _ in range(num_balloons)]
    
    for i in range(num_balloons):
        balloon_data = data[:, i, :][::-1]  # Reverse order
        balloon_data = scalers[i].fit_transform(balloon_data)
        
        for j in range(len(balloon_data) - time_steps):
            X.append(balloon_data[j:j+time_steps])
            y.append(balloon_data[j+time_steps])
    
    return np.array(X), np.array(y), scalers

# Train LSTM
def retrain_model(data):
    X, y, _ = process_sequences(data)
    model.fit(X, y, epochs=30, batch_size=16, verbose=1)

# Predict next positions
def predict_next_positions(data, time_steps=5):
    X, _, scalers = process_sequences(data, time_steps)
    X = X[-1].reshape(1, time_steps, 3)
    prediction_scaled = model.predict(X)
    prediction_real = [scalers[i].inverse_transform(prediction_scaled.reshape(-1, 3)) for i in range(len(scalers))]
    return prediction_real

@app.post("/predict")
def predict(request: PredictionRequest):
    openai_api_key = request.openai_api_key
    if not openai_api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key required")

    data = fetch_windborne_data()
    if data is None:
        raise HTTPException(status_code=500, detail="No data available")

    retrain_model(data)
    prediction = predict_next_positions(data)

    # Generate LLM analysis
    template = """You are an operational analyst for a weather balloon company. Analyze the data of 
    balloon positions over 24H: {all_positions}, LSTM prediction: {LSTM_prediction}, pred next position: {pred_next_position}
    
    Extract 3 key insights, such as:
    - Spatial clusters indicating wind patterns
    - Anomalies (e.g., balloons stuck in one area)
    - Suggestions for optimizing future launches
    Format the response as a bullet-point report."""
    
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4-turbo")
    chain = prompt | model
    report = chain.invoke({
        "all_positions": str(data), 
        "LSTM_prediction": str(prediction), 
        "pred_next_position": str(prediction)
    })

    return {"prediction": prediction, "insights": report.content}