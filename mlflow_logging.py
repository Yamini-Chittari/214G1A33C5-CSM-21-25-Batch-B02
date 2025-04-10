import mlflow
import mlflow.sklearn
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score

def log_model(model, model_name, model_path):
    """
    Log the trained model to MLflow and save it locally.
    """
    with mlflow.start_run(run_name=f"Logging {model_name}"):
        mlflow.sklearn.log_model(model, artifact_path=model_name)
        joblib.dump(model, model_path)
        print(f"Model '{model_name}' logged and saved successfully at '{model_path}'.")

def log_training_metrics(y_true, y_pred, model_name, task_type="classification"):
    """
    Log training metrics to MLflow.
    - For classification: logs accuracy.
    - For regression: logs mean squared error (MSE).
    """
    with mlflow.start_run(run_name=f"Metrics for {model_name}"):
        if task_type == "classification":
            accuracy = accuracy_score(y_true, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            print(f"Accuracy logged: {accuracy}")
        elif task_type == "regression":
            mse = mean_squared_error(y_true, y_pred)
            mlflow.log_metric("mean_squared_error", mse)
            print(f"Mean Squared Error logged: {mse}")

def log_prediction(input_data, prediction, model_name):
    """
    Log prediction results to MLflow.
    """
    with mlflow.start_run(run_name=f"Prediction using {model_name}"):
        # Convert input data to dictionary format for logging
        mlflow.log_param("Temperature (C)", input_data['Temperature (°C)'][0])
        mlflow.log_param("Pressure (bar)", input_data['Pressure (bar)'][0])
        mlflow.log_param("Vibration (mm/s)", input_data['Vibration (mm/s)'][0])
        mlflow.log_param("Working Hours", input_data['Working_Hours'][0])

        # Log the prediction output
        mlflow.log_param("Failure Type", prediction['failure_type'])
        mlflow.log_param("Flight Type", prediction['flight_type'])
        mlflow.log_param("Flight Number", prediction['flight_number'])
        mlflow.log_param("Remaining Operational Hours", prediction['predicted_lifespan'])
        mlflow.log_param("Recommendation", prediction['recommendation'])

        print(f"Prediction logged successfully for '{model_name}'.")
