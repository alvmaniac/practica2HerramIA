import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path):
    # df = pd.read_csv(file_path)
    df =pd.read_csv(file_path, delimiter=',')

    # Eliminar filas con valores nulos
    df.dropna(inplace=True)

    # Codificar variables categóricas
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

    X = df.drop(columns=["Avenue"])
    y = df["Avenue"]

    return X, y, label_encoders
