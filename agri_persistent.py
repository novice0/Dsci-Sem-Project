import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv("agriculture.csv")

# Preprocess the data
df = df.drop(
    columns=[
        "Domain Code",
        "Domain",
        "Area Code (M49)",
        "Area",
        "Year Code",
        "Year",
        "Source Code",
        "Source",
        "Unit",
        "Flag",
        "Flag Description",
        "Note",
    ]
)

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=["object"]).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Define features and target variable
X = df.drop(columns=["Element"])
y = df["Element"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the trained model
with open("random_forest_model.pkl", "wb") as model_file:
    pickle.dump(clf, model_file)

# Save the label encoders
with open("label_encoders.pkl", "wb") as encoder_file:
    pickle.dump(label_encoders, encoder_file)
