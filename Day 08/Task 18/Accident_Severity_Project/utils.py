import pickle
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "Road Accident Data.csv"

PREFERRED_MONTH_ORDER = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]
PREFERRED_DAY_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

VISIBLE_FIELDS = {
    "month": "Month",
    "day_of_week": "Day_of_Week",
    "junction_control": "Junction_Control",
    "junction_detail": "Junction_Detail",
    "light_conditions": "Light_Conditions",
    "road_surface_conditions": "Road_Surface_Conditions",
    "road_type": "Road_Type",
    "weather_conditions": "Weather_Conditions",
    "vehicle_type": "Vehicle_Type",
    "urban_or_rural_area": "Urban_or_Rural_Area",
}

NUMERIC_FIELDS = {
    "speed_limit": "Speed_limit",
    "vehicles": "Number_of_Vehicles",
    "casualties": "Number_of_Casualties",
}

HIDDEN_FIELD_DEFAULTS = {
    "Accident_Index": None,
    "Accident Date": None,
    "Year": None,
    "Latitude": None,
    "Longitude": None,
    "Local_Authority_(District)": None,
    "Carriageway_Hazards": None,
    "Police_Force": None,
    "Time": None,
}


def _load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.fillna(df.mode().iloc[0])
    return df


def _build_encoders(df: pd.DataFrame) -> dict:
    encoders = {}
    object_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in object_columns:
        encoder = LabelEncoder()
        encoder.fit(df[col].astype(str))
        encoders[col] = encoder
    return encoders


def _sort_values(column_name: str, values: pd.Index) -> list:
    if column_name == "Month":
        return [m for m in PREFERRED_MONTH_ORDER if m in values]
    if column_name == "Day_of_Week":
        return [d for d in PREFERRED_DAY_ORDER if d in values]
    return sorted(values)


df = _load_dataset()
encoders = _build_encoders(df)
model = None
model_accuracy = None


def load_model() -> object:
    global model
    if model is None:
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
    return model


def get_model_accuracy() -> float:
    global model_accuracy
    if model_accuracy is None:
        model_obj = load_model()
        X = df.drop("Accident_Severity", axis=1).copy()
        for col in X.select_dtypes(include=["object", "string"]).columns.tolist():
            X[col] = encoders[col].transform(X[col].astype(str))
        y = encoders["Accident_Severity"].transform(df["Accident_Severity"].astype(str))
        model_accuracy = float(model_obj.score(X, y))
    return model_accuracy


def get_form_options() -> dict:
    options = {}
    for key, column in VISIBLE_FIELDS.items():
        values = df[column].dropna().astype(str).unique().tolist()
        options[key] = _sort_values(column, values)
    options["urban_or_rural_area"] = _sort_values(
        VISIBLE_FIELDS["urban_or_rural_area"],
        df[VISIBLE_FIELDS["urban_or_rural_area"]].dropna().astype(str).unique()
    )
    return options


def get_default_values() -> dict:
    return {
        "month": "Jan",
        "day_of_week": "Monday",
        "junction_control": df["Junction_Control"].dropna().astype(str).mode().iloc[0],
        "junction_detail": df["Junction_Detail"].dropna().astype(str).mode().iloc[0],
        "light_conditions": df["Light_Conditions"].dropna().astype(str).mode().iloc[0],
        "road_surface_conditions": df["Road_Surface_Conditions"].dropna().astype(str).mode().iloc[0],
        "road_type": df["Road_Type"].dropna().astype(str).mode().iloc[0],
        "weather_conditions": df["Weather_Conditions"].dropna().astype(str).mode().iloc[0],
        "vehicle_type": df["Vehicle_Type"].dropna().astype(str).mode().iloc[0],
        "urban_or_rural_area": df["Urban_or_Rural_Area"].dropna().astype(str).mode().iloc[0],
        "speed_limit": int(df["Speed_limit"].median()),
        "vehicles": int(df["Number_of_Vehicles"].median()),
        "casualties": int(df["Number_of_Casualties"].median()),
    }


def _build_input_row(form: dict) -> dict:
    row = {
        "Accident_Index": df["Accident_Index"].dropna().astype(str).iloc[0],
        "Accident Date": df["Accident Date"].dropna().astype(str).iloc[0],
        "Month": form["month"],
        "Day_of_Week": form["day_of_week"],
        "Year": int(df["Year"].mode().iloc[0]),
        "Junction_Control": form["junction_control"],
        "Junction_Detail": form["junction_detail"],
        "Latitude": float(df["Latitude"].median()),
        "Light_Conditions": form["light_conditions"],
        "Local_Authority_(District)": df["Local_Authority_(District)"].dropna().astype(str).mode().iloc[0],
        "Carriageway_Hazards": df["Carriageway_Hazards"].dropna().astype(str).mode().iloc[0],
        "Longitude": float(df["Longitude"].median()),
        "Number_of_Casualties": int(form["casualties"]),
        "Number_of_Vehicles": int(form["vehicles"]),
        "Police_Force": df["Police_Force"].dropna().astype(str).mode().iloc[0],
        "Road_Surface_Conditions": form["road_surface_conditions"],
        "Road_Type": form["road_type"],
        "Speed_limit": int(form["speed_limit"]),
        "Time": df["Time"].dropna().astype(str).mode().iloc[0],
        "Urban_or_Rural_Area": form["urban_or_rural_area"],
        "Weather_Conditions": form["weather_conditions"],
        "Vehicle_Type": form["vehicle_type"],
    }
    return row


def encode_input(row: dict) -> dict:
    encoded = {}
    for key, value in row.items():
        if key in encoders:
            encoded[key] = int(encoders[key].transform([str(value)])[0])
        else:
            encoded[key] = value
    return encoded


def predict_severity(form: dict) -> dict:
    model_obj = load_model()
    row = _build_input_row(form)
    encoded_row = encode_input(row)
    df_input = pd.DataFrame([encoded_row])

    prediction = model_obj.predict(df_input)[0]
    probabilities = model_obj.predict_proba(df_input)[0]

    target_encoder = encoders["Accident_Severity"]
    predicted_label = target_encoder.inverse_transform([prediction])[0]

    classes = [target_encoder.inverse_transform([c])[0] for c in model_obj.classes_]
    prob_list = sorted(
        [(label, float(prob)) for label, prob in zip(classes, probabilities)],
        key=lambda item: item[1],
        reverse=True
    )

    confidence = float(probabilities[list(model_obj.classes_).index(prediction)])
    if confidence >= 0.8:
        confidence_text = "High confidence"
    elif confidence >= 0.6:
        confidence_text = "Moderate confidence"
    else:
        confidence_text = "Low confidence"

    max_prob = float(max(probabilities))
    model_acc = get_model_accuracy()
    if max_prob <= 0.4:
        impact_text = "Slight impact"
    elif max_prob <= 0.7:
        impact_text = "Medium impact"
    else:
        impact_text = "Fatal severity"

    return {
        "label": predicted_label,
        "probability": max_prob,
        "impact_label": impact_text,
        "accuracy": model_acc,
        "confidence_text": confidence_text,
        "probabilities": prob_list,
        "risk_message": (
            "The model indicates the incident is likely to be severe. "
            if predicted_label in ["Serious", "Fatal"] else
            "The model indicates a lower severity outcome, but continue to take precautions."
        )
    }
