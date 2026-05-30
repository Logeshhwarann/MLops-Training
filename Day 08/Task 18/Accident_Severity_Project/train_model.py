import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("data/Road Accident Data.csv", nrows=100000)

# ==========================================
# REMOVE MISSING VALUES
# ==========================================

for col in df.columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# ==========================================
# ENCODE CATEGORICAL COLUMNS
# ==========================================

le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df.drop("Accident_Severity", axis=1)

y = df["Accident_Severity"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# ACCURACY
# ==========================================

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("Model Accuracy:", accuracy)

# ==========================================
# SAVE MODEL
# ==========================================

with open("models/best_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")