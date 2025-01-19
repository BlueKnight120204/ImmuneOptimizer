import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import random

# Load Datasets
genetic_data = pd.read_csv("Genetic_dataset.csv")
nutrition_data = pd.read_csv("Nutrition_Dataset.csv")

# Step 1: Generate Recommended Nutrients
def generate_refined_recommendations(genetic_data, nutrition_data):
    recommendations = []

    # Define standard ranges for nutrients
    nutrient_ranges = {
        "Protein": (40, 60),
        "Fat": (20, 35),
        "Carbohydrates": (200, 300),
        "Dietary Fiber": (25, 38),
        "Vitamin A": (700, 900),
        "Vitamin C": (65, 90),
        "Vitamin D": (15, 20),
        "Zinc": (8, 11),
        "Iron": (8, 18),
        "Calcium": (1000, 1300),
        "Selenium": (55, 70),
        "Water": (2.7, 3.7),  # Based on hydration status
    }

    # Iterate through each individual in the genetic dataset
    for index, row in genetic_data.iterrows():
        gender = row["Gender"]
        bmi = row["BMI"]
        hydration_status = row["Hydration_Status"]

        # Initialize recommended nutrients with variations
        recommended_nutrients = {}
        for nutrient, (min_value, max_value) in nutrient_ranges.items():
            # Adjust values based on individual factors with randomness
            if nutrient == "Protein":
                base_value = max_value if gender == "Male" else min_value
                variation = random.uniform(-5, 5)
                value = base_value + variation
            elif nutrient == "Fat":
                base_value = min_value if bmi > 25 else max_value
                variation = random.uniform(-3, 3)
                value = base_value + variation
            elif nutrient == "Carbohydrates":
                base_value = max_value if bmi < 18.5 else min_value
                variation = random.uniform(-10, 10)
                value = base_value + variation
            elif nutrient == "Water" and hydration_status == "Insufficient":
                value = max_value
            else:
                base_value = (min_value + max_value) / 2
                variation = random.uniform(-(max_value - min_value) * 0.1, (max_value - min_value) * 0.1)
                value = base_value + variation

            # Ensure the value stays within the min and max bounds
            value = max(min_value, min(max_value, value))

            # Add nutrient recommendation
            recommended_nutrients[nutrient] = round(value, 1)

        # Append the individual's recommendations
        recommendations.append(recommended_nutrients)

    return pd.DataFrame(recommendations)

print("Generating nutrient recommendations...")
recommended_nutrients_df = generate_refined_recommendations(genetic_data, nutrition_data)
genetic_data_with_nutrients = pd.concat([genetic_data, recommended_nutrients_df], axis=1)

# Step 3: Encode Categorical Columns
categorical_columns = [
    "Gender", 
    "Dietary_Preferences", 
    "Alcohol_Consumption", 
    "Smoking_Status", 
    "Hydration_Status", 
    "Case_Label", 
    "Physical_Activity", 
    "Sleep_Quality"
]

print("Encoding categorical columns...")
for col in categorical_columns:
    genetic_data_with_nutrients[col] = genetic_data_with_nutrients[col].astype("category").cat.codes

# Step 4: Prepare Features and Targets
feature_columns = genetic_data.columns
target_columns = list(recommended_nutrients_df.columns)

X = genetic_data_with_nutrients[feature_columns]
y = genetic_data_with_nutrients[target_columns]

# Step 5: Train-Test Split and Model Training
print("Splitting data and training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Model Evaluation
print("Evaluating model...")
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 7: Save the Model and Nutrition Data
print("Saving the model and nutrition data...")
model_and_data = {
    "model": model,
    "nutrition_data": nutrition_data
}

joblib.dump(model_and_data, "PNSmodel.pkl")
print("Model saved as 'PNSmodel.pkl'.")