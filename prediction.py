import random
import joblib
import pandas as pd
from model.recommendation import generate_recommendation
from mlflow_utils.mlflow_logging import log_prediction

# Load the models
maintenance_model = joblib.load('models/maintenance_classifier_model.pkl')
lifespan_model = joblib.load('models/remaining_life_regressor_model.pkl')


def make_predictions(input_data):
    """
    Makes predictions using pre-trained models and logs them to MLflow.

    Args:
        input_data (pd.DataFrame): DataFrame containing input features.

    Returns:
        str: Generated recommendation.
    """
    X_input = input_data[['Temperature (°C)', 'Pressure (bar)', 'Vibration (mm/s)', 'Working_Hours']]

    # Predict maintenance and remaining lifespan
    predicted_maintenance = maintenance_model.predict(X_input)[0]
    predicted_lifespan = lifespan_model.predict(X_input)[0]

    # Generate failure type and flight recommendation
    failure_type = "None" if predicted_maintenance == 0 else random.choice(
        ['Turbine Overheat', 'Pressure Loss', 'Valve Stuck']
    )
    flight_type = random.choice(['commercial', 'cargo', 'private', 'long-haul', 'short-haul'])
    flight_number = random.randint(1000, 9999)

    recommendation = generate_recommendation(failure_type, flight_type, flight_number, predicted_lifespan)

    # Log the prediction to MLflow
    log_prediction(
        temperature=X_input['Temperature (°C)'][0],
        pressure=X_input['Pressure (bar)'][0],
        vibration=X_input['Vibration (mm/s)'][0],
        working_hours=X_input['Working_Hours'][0],
        predicted_maintenance=predicted_maintenance,
        predicted_lifespan=predicted_lifespan,
        recommendation=recommendation
    )

    return recommendation
