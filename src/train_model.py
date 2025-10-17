import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib

# Opcional: para visualizar la matriz de confusión
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_save_model(input_path):
    df = pd.read_csv(input_path)

    print("Valores únicos en 'label' antes de limpiar:", df['label'].unique())
    print("Conteo de 'label' antes de limpiar:")
    print(df['label'].value_counts())

    # Eliminar filas donde 'label' es NaN
    df = df.dropna(subset=['label'])

    # Asegurarse de que la columna 'label' esté en formato numérico
    df['label_num'] = df['label'].map({'legitimate': 0, 'phishing': 1})

    # Verificar que no haya NaN en label_num después del mapeo
    df = df.dropna(subset=['label_num'])
    df['label_num'] = df['label_num'].astype(int)  # Asegurar tipo entero

    print("\nValores únicos en 'label_num' después de limpiar y mapear:", df['label_num'].unique())
    print("Conteo de 'label_num' después de limpiar y mapear:")
    print(df['label_num'].value_counts())

    # Separar características (X) y etiquetas (y)
    X = df.drop(columns=['url', 'label', 'label_num'])
    y = df['label_num']

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === Entrenar modelo (puedes cambiar por XGBoost o Regresión Logística) ===
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    # === Predecir y evaluar ===
    y_pred = model.predict(X_test)

    print("\nMétricas del modelo XGBoost:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-Score:", f1_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # === Visualizar la matriz de confusión y guardarla ===
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusión")
    plt.ylabel("Real")
    plt.xlabel("Predicción")
    plt.savefig("confusion_matrix.png")  # Guardar la imagen
    # plt.show()  # Comentar o eliminar esta línea
    print("Matriz de confusión guardada como confusion_matrix.png")

    # === Guardar modelo ===
    joblib.dump(model, "modelo_phishing_detector.pkl")
    print("\nModelo guardado como modelo_phishing_detector.pkl")

if __name__ == "__main__":
    train_and_save_model("./data/processed_data.csv")