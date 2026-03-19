from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd
import tf_keras as keras

app = FastAPI(title="Salary Prediction API")

model = keras.models.load_model("reg-model.h5")

with open("label_encode_gen.pkl", "rb") as f:
    label_encode_gen = pickle.load(f)

with open("onehot_encode_geo.pkl", "rb") as f:
    onehot_encode_geo = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

class CustomerInput(BaseModel):
    CreditScore:      int
    Geography:        str
    Gender:           str
    Age:              int
    Tenure:           int
    Balance:          float
    NumOfProducts:    int
    HasCrCard:        int
    IsActiveMember:   int
    Exited:           int

@app.post("/predict")
def predict_salary(data: CustomerInput):
    gender_encoded = label_encode_gen.transform([data.Gender])[0]

    geo_df = pd.DataFrame([[data.Geography]], columns=["Geography"])
    geo_encoded = onehot_encode_geo.transform(geo_df).toarray()
    geo_df_enc = pd.DataFrame(geo_encoded, columns=onehot_encode_geo.get_feature_names_out(["Geography"]))

    input_df = pd.DataFrame([{
        "CreditScore":    data.CreditScore,
        "Gender":         gender_encoded,
        "Age":            data.Age,
        "Tenure":         data.Tenure,
        "Balance":        data.Balance,
        "NumOfProducts":  data.NumOfProducts,
        "HasCrCard":      data.HasCrCard,
        "IsActiveMember": data.IsActiveMember,
        "Exited":         data.Exited,
    }])

    input_df = pd.concat([input_df.reset_index(drop=True), geo_df_enc.reset_index(drop=True)], axis=1)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    return {"predicted_salary": round(float(prediction[0][0]), 2)}

@app.get("/")
def root():
    return {"message": "Salary Prediction API is running!"}

