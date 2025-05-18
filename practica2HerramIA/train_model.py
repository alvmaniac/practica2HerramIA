import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_preprocessing import load_and_preprocess_data
import joblib

mlflow.set_tracking_uri("http://localhost:9090")
mlflow.set_experiment("Predicción de Inverción Financiera")

if __name__ == "__main__":
    X, y, label_encoders = load_and_preprocess_data("finance_data.csv")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        
          # Guardar el run_id en un archivo
        run_id = run.info.run_id
        with open("latest_run_id.txt", "w") as f:
            f.write(run_id)

        print(f"Modelo entrenado con precisión: {accuracy}")
        print(f"Run ID guardado en latest_run_id.txt: {run_id}")
