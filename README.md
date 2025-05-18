Modelo para Predecir la elección de inversión de los clientes

1. **Inicia MLflow UI:**

mlflow ui --port 9090

2. **Ejecuta el entrenamiento y despliegue del modelo:**

python train_model.py

-- El modelo se depliega en http://loclahost:9090

3. **Ejecuta el servicio de predicción:**

python predict_service.py

4. **Predecir Data Input (Postman):**

- URI: http://localhost:9091/predict
- Método: Post
- Body:

{
    "gender": "Female",
    "age": 22,
    "Investment_Avenues": "No",
    "Mutual_Funds": 2,
    "Equity_Market": 6,
    "Debentures": 4,
    "Government_Bonds": 2,
    "Fixed_Deposits": 5,
    "PPF": 1,
    "Gold": 7,
    "Stock_Marktet": "No",
    "Factor": "Returns",
    "Objective": "Capital Appreciation",
    "Purpose": "Wealth Creation",
    "Duration": "3-5 years",
    "Invest_Monitor": "Daily",
    "Expect": "20%-30%",
    "What are your savings objectives?": "Retirement Plan",
    "Reason_Equity": "Capital Appreciation",
    "Reason_Mutual": "Tax Benefits",
    "Reason_Bonds": "Assured Returns",
    "Reason_FD": "Fixed Returns",
    "Source": "Television"
  }

4. **Ejecución:**
![image](https://github.com/user-attachments/assets/c0da41a8-769a-49bf-bfc7-b2f4c1d06e4e)
