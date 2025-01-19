from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd
import gdown

app = Flask(__name__)

# Load the saved model and nutrition data
file_id = '1fQ3rKN139TNPNGXgdxKqJC1_HwQXLyHc'
url = f'https://drive.google.com/uc?export=download&id={file_id}'
gdown.download(url, 'PNS_model.pkl', quiet=False)

loaded_data = joblib.load("PNS_model.pkl")
model = loaded_data["model"]

nutrition_data = pd.read_csv("Nutrition_Dataset.csv")

# Function to calculate Hydration Status based on water intake
def calculate_hydration_status(water_intake):
    return 'Sufficient' if water_intake >= 2 else 'Insufficient'

# Function to determine Case_Label
def determine_case_label(age, bmi):
    if age > 50 and bmi > 30:
        return 'Autoimmune Risk'
    elif bmi < 18.5:
        return 'Nutrient Metabolism'
    return 'Inflammation'

# Preprocess input (consistent with training)
def preprocess_input(input_data):
    # Encode categorical variables    
    categorical_mappings = {
        'Gender': {'Male': 1, 'Female': 0},
        'Dietary_Preferences': {'Vegetarian': 0, 'Non-Vegetarian': 1},
        'Alcohol_Consumption': {'Never': 0, 'Occasionally': 1, 'Frequently': 2},
        'Smoking_Status': {'Non-Smoker': 0, 'Smoker': 1},
        'Physical_Activity': {'Sedentary': 0, 'Moderate': 1, 'Active': 2},
        'Sleep_Quality': {'Poor': 0, 'Average': 1, 'Good': 2},
        'Case_Label': {'Inflammation': 0, 'Autoimmune Risk': 1, 'Nutrient Metabolism': 2},
        'Hydration_Status': {'Sufficient': 1, 'Insufficient': 0},
    }
    
    # Apply mappings to categorical features
    for feature, mapping in categorical_mappings.items():
        input_data[feature] = mapping[input_data[feature]]
    
    # Ensure the order of features matches the training dataset (93 features)    
    feature_order = [f"SNP_{i+1}" for i in range(75)] + [
        'Nutrient_1', 'Nutrient_2', 'Nutrient_3', 'Nutrient_4', 'Nutrient_5', 'Nutrient_6',
        'Age', 'Gender', 'BMI', 'Physical_Activity', 'Sleep_Quality', 'Dietary_Preferences',
        'Alcohol_Consumption','Smoking_Status', 'Exercise_Hours_Per_Week', 'Water_Intake_Liters',
        'Hydration_Status', 'Case_Label'
    ]
    
    # Ensure all keys are present in input_data
    for feature in feature_order:
        if feature not in input_data:
            input_data[feature] = 0  # Fill with 0 for missing features
    
    processed_input = [input_data[feature] for feature in feature_order]
    return np.array(processed_input).reshape(1, -1)

# Categorization logic based on user-provided draft
def categorize_foods(nutrition_data):
    # Define categories with their keywords
    categories = {
        'Meat & Poultry': [
            'beef', 'chicken', 'pork', 'lamb', 'turkey', 'veal', 'duck', 'goose',
            'pheasant', 'emu', 'ostrich', 'bacon', 'ham', 'salami', 'sausage',
            'pastrami', 'bologna', 'mutton'
        ],
        'Seafood': [
            'fish', 'salmon', 'tuna', 'cod', 'halibut', 'crab', 'shrimp', 'lobster',
            'oyster', 'clam', 'mussel', 'octopus', 'squid', 'anchovy', 'catfish',
            'trout', 'abalone', 'caviar', 'eel'
        ],
        'Dairy & Eggs': [
            'cheese', 'milk', 'yogurt', 'cream', 'butter', 'egg', 'cottage cheese',
            'mozzarella', 'cheddar', 'parmesan', 'brie', 'feta', 'gouda'
        ],
        'Fruits': [
            'apple', 'banana', 'orange', 'grape', 'pear', 'peach', 'plum', 'berry',
            'melon', 'cherry', 'kiwi', 'mango', 'pineapple', 'fig', 'date',
            'papaya', 'apricot', 'lemon', 'lime', 'grapefruit'
        ],
        'Vegetables': [
            'carrot', 'potato', 'tomato', 'onion', 'lettuce', 'cucumber', 'pepper',
            'broccoli', 'spinach', 'cabbage', 'celery', 'asparagus', 'corn',
            'pea', 'bean', 'squash', 'zucchini', 'eggplant', 'artichoke'
        ],
        'Grains & Cereals': [
            'rice', 'wheat', 'oat', 'barley', 'quinoa', 'corn', 'rye', 'millet',
            'buckwheat', 'cereal', 'pasta', 'noodle', 'bread', 'flour', 'couscous'
        ],
        'Nuts & Seeds': [
            'almond', 'peanut', 'cashew', 'walnut', 'pistachio', 'pecan',
            'macadamia', 'hazelnut', 'seed', 'pine nut', 'chestnut'
        ],
        'Legumes & Pulses': [
            'bean', 'lentil', 'pea', 'chickpea', 'soybean', 'fava', 'kidney',
            'black bean', 'navy bean', 'pinto bean'
        ],
        'Beverages': [
            'juice', 'coffee', 'tea', 'milk', 'water', 'cocktail', 'smoothie'
        ],
        'Sweets & Desserts': [
            'candy', 'chocolate', 'ice cream', 'cookie', 'cake', 'pie', 'pudding',
            'brownie', 'pastry', 'gelatin', 'marshmallow'
        ],
        'Mushrooms & Fungi': [
            'mushroom', 'truffle', 'morel', 'chanterelle', 'shiitake', 'porcini',
            'enoki', 'fungus'
        ],
        'Prepared Dishes': [
            'soup', 'stew', 'casserole', 'salad', 'curry', 'pasta'
        ]
    }
    
    # Assign categories to foods based on keywords
    nutrition_data["Category"] = "Others"  # Default category
    for category, keywords in categories.items():
        nutrition_data.loc[
            nutrition_data["food"].str.contains("|".join(keywords), case=False, na=False),
            "Category"
        ] = category

    return nutrition_data

# Refine healthy/unhealthy classification logic
def classify_healthiness(nutrition_data):
    # Example thresholds tailored to the dataset
    healthy_thresholds = {
        "Fat": 15,  # Low fat
        "Carbohydrates": 50,  # Moderate carbs
        "Protein": 10,  # High protein
        "Sodium": 200,  # Low sodium
        "Sugar": 10  # Low sugar
    }

    # Add a Healthiness column based on thresholds
    nutrition_data["Healthiness"] = "Healthy"  # Default to Healthy
    for col, threshold in healthy_thresholds.items():
        if col in nutrition_data.columns:
            nutrition_data.loc[nutrition_data[col] > threshold, "Healthiness"] = "Unhealthy"

    return nutrition_data

def generate_refined_food_recommendations_with_predicted_nutrients(predicted_nutrients, nutrition_data):
    # Categorize and classify food healthiness
    categorized_data = categorize_foods(nutrition_data)
    classified_data = classify_healthiness(categorized_data)
    
    # Nutrient columns from the model
    nutrient_columns = ['Protein', 'Fat', 'Carbohydrates', 'Dietary Fiber', 'Vitamin A', 
                        'Vitamin C', 'Vitamin D', 'Zinc', 'Iron', 'Calcium', 'Selenium']

    general_fallback_foods = ["apple", "banana", "carrot", "spinach", "chicken", "salmon", "oats", "almonds", "yogurt"]

    recommendations = []

    # Loop through each nutrient and generate recommendations
    for nutrient, predicted_value in zip(nutrient_columns, predicted_nutrients):
        recommended_foods = []

        # Find foods rich in the predicted nutrient
        nutrient_data = classified_data[
            (classified_data[nutrient] >= predicted_value) & (classified_data["Healthiness"] == "Healthy")
        ]
        
        # Sort by nutrient values in descending order and extract top foods
        nutrient_data_sorted = nutrient_data.sort_values(by=nutrient, ascending=False)

        # Get top foods, making sure to collect diverse categories
        top_foods = []
        for category in nutrient_data_sorted["Category"].unique():
            category_foods = nutrient_data_sorted[nutrient_data_sorted["Category"] == category]
            top_foods.extend(category_foods["food"].head(2).tolist())  # Limit to 2 foods per category
            if len(top_foods) >= 5:  # Limit to 5 total recommendations
                break
        
        # If no foods are found, use general fallback foods
        if not top_foods:
            top_foods = general_fallback_foods[:5]  # Provide fallback options

        # Add nutrient and its recommended foods to final output
        recommendations.append({
            "Nutrient": nutrient,
            "Recommended Foods": top_foods
        })
    
    return recommendations


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    data['Hydration_Status'] = calculate_hydration_status(data['Water_Intake_Liters'])
    data['Case_Label'] = determine_case_label(data['Age'], data['BMI'])
    processed_input = preprocess_input(data)
    predicted_nutrients = model.predict(processed_input)[0]
    recommendations = generate_refined_food_recommendations_with_predicted_nutrients(predicted_nutrients, nutrition_data)
    return jsonify({"Predicted Nutrients": predicted_nutrients.tolist(), "Recommendations": recommendations})

if __name__ == "__main__":
    app.run(debug=True)