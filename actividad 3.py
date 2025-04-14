# -*- coding: utf-8 -*-
"""Modelo de árbol de decisión para predecir demanda en transporte masivo"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# ======================
# 1. GENERAR DATASET (Si no tienes datos reales)
# ======================
np.random.seed(42)  # Para reproducibilidad

# Crear datos ficticios
data = {
    'Hora_pico': np.random.choice(['Sí', 'No'], 1000, p=[0.6, 0.4]),
    'Día_semana': np.random.choice(['Laboral', 'Fin de semana'], 1000, p=[0.7, 0.3]),
    'Clima': np.random.choice(['Lluvia', 'Soleado', 'Nublado'], 1000, p=[0.3, 0.5, 0.2]),
    'Demanda_alta': np.random.choice(['Sí', 'No'], 1000, p=[0.65, 0.35])  # Target
}

df = pd.DataFrame(data)

# Guardar dataset (opcional)
df.to_csv('dataset_transporte_masivo.csv', index=False)

# ======================
# 2. PREPROCESAMIENTO
# ======================
# Codificar variables categóricas a numéricas
df_encoded = pd.get_dummies(df, columns=['Hora_pico', 'Día_semana', 'Clima'])

# Separar features (X) y target (y)
X = df_encoded.drop('Demanda_alta', axis=1)
y = df_encoded['Demanda_alta']

# Dividir en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ======================
# 3. ENTRENAR MODELO
# ======================
modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
modelo.fit(X_train, y_train)

# ======================
# 4. EVALUAR MODELO
# ======================
# Predecir sobre test
y_pred = modelo.predict(X_test)

# Métricas
precision = accuracy_score(y_test, y_pred)
matriz_confusion = confusion_matrix(y_test, y_pred)

print(f"\nPrecisión del modelo: {precision:.2%}")
print("\nMatriz de confusión:")
print(matriz_confusion)

# ======================
# 5. VISUALIZACIÓN
# ======================
# Árbol de decisión
plt.figure(figsize=(12, 8))
plot_tree(modelo, 
          feature_names=X.columns, 
          class_names=['No', 'Sí'], 
          filled=True, 
          rounded=True)
plt.title("Árbol de Decisión - Predicción de Demanda Alta")
plt.show()

# Importancia de características
importancias = modelo.feature_importances_
features_importantes = pd.DataFrame({
    'Feature': X.columns,
    'Importancia': importancias
}).sort_values('Importancia', ascending=False)

print("\nImportancia de cada feature:")
print(features_importantes)