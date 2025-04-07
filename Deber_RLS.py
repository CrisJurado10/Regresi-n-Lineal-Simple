# Regresión Lineal Simple
#Link del CSV obtenido: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=train.csv
#Se utilizó el dataset train.csv de la competencia de Kaggle "House Prices: Advanced Regression Techniques", con más de 1400 registros. 
# Se aplicó Regresión Lineal Simple para modelar la relación entre el área habitable (GrLivArea) y el precio de venta (SalePrice).
# Importar las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('train.csv')

# Usar nombres de columnas 
X = dataset[['GrLivArea']].values  # Nota: usamos doble corchete para que X sea matriz (2D)
y = dataset['SalePrice'].values  # Esto puede ser 1D

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Entrenar el modelo de Regresión Lineal Simple
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predecir los resultados del conjunto de prueba
y_pred = regressor.predict(X_test)

# Visualizar los resultados del conjunto de entrenamiento
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Precio de Venta vs Área Habitable (Entrenamiento)')
plt.xlabel('Área Habitable (GrLivArea)')
plt.ylabel('Precio de Venta (SalePrice)')
plt.show()

# Visualizar los resultados del conjunto de prueba
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')  # Misma línea que en entrenamiento
plt.title('Precio de Venta vs Área Habitable (Prueba)')
plt.xlabel('Área Habitable (GrLivArea)')
plt.ylabel('Precio de Venta (SalePrice)')
plt.show()

#Interpretaciíon 
# Análisis Matemático de la Regresión Lineal Simple:
# El modelo entrenado es de la forma: y = β₀ + β₁ * x
# Variable Dependiente: SalePrice
# Variable Independiente: GrLivArea
# Donde:
#   - y es el Precio de Venta (SalePrice)
#   - x es el Área Habitable (GrLivArea)
#   - β₀ es la ordenada al origen (intercepto)
#   - β₁ es la pendiente (coeficiente de regresión)

# La regresión lineal simple estima los coeficientes β₀ y β₁

# Significado de los coeficientes:
#    - β₁ (pendiente): representa el cambio estimado en el precio de venta por cada unidad adicional 
#      de área habitable. Si β₁ > 0, la relación es positiva.
#    - β₀ (intercepto): representa el valor estimado del precio de venta cuando el área habitable es 0.

# En este caso, observamos una correlación positiva entre GrLivArea y SalePrice,
# lo cual se visualiza en la pendiente positiva de la línea azul.

# Análisis de las gráficas
# 1. Gráfica del conjunto de ENTRENAMIENTO:
#    - Los puntos rojos representan los datos reales de entrenamiento (SalePrice vs GrLivArea).
#    - La línea azul es la línea de regresión ajustada: y = β₀ + β₁ * x.
#    - Se observa una clara correlación positiva: a mayor área habitable, mayor es el precio de venta.
#    - Esto se traduce en una pendiente positiva β₁ > 0.
#    - La mayoría de los puntos están cercanos a la línea, lo cual indica un buen ajuste en el conjunto de entrenamiento.
#    - Sin embargo, hay outliers con grandes desviaciones, lo que sugiere que otros factores también influyen en el precio.

# 2. Gráfica del conjunto de PRUEBA:
#    - Se evalúa qué tan bien generaliza el modelo a datos no vistos.
#    - La línea azul es la misma que en la gráfica anterior (misma recta ajustada).
#    - Los puntos rojos son los valores reales de prueba.
#    - Se observan patrones similares: alineación general con la recta, pero con mayor dispersión que en entrenamiento.
#    - Esto sugiere que el modelo tiene una capacidad aceptable de generalización, pero no capta toda la complejidad.
#    - La variabilidad no explicada por la línea indica que una sola variable (GrLivArea) no es suficiente para predecir con alta precisión.


# En resumen:
# El modelo captura la tendencia general del crecimiento de precios con respecto al área habitable,
# La dispersión de los puntos sugiere que una regresión lineal simple puede no ser suficiente para explicar completamente la variabilidad en los precios de venta
# La regresión lineal simple proporciona una primera aproximación útil a la relación entre el área habitable
# El precio de venta, pero su poder predictivo es limitado debido a la naturaleza multivariable del problema.



