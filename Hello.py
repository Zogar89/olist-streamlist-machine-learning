import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import datetime

# Función para descargar archivos
def download_file(url, file_name):
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_name, 'wb') as file:
            file.write(response.content)
        return os.path.abspath(file_name)
    else:
        return f"Error al descargar el archivo: Código de estado {response.status_code}"

# URLs de los archivos
order_items_url = "https://github.com/MorenaCaparros/ScrappingProyectoGrupal/raw/main/e-comerce_Olist_dataset/olist_order_items_dataset.csv"
orders_url = "https://github.com/MorenaCaparros/ScrappingProyectoGrupal/raw/main/e-comerce_Olist_dataset/olist_orders_dataset.csv"

# Nombres de los archivos descargados
order_items_file_name = "olist_order_items_dataset.csv"
orders_file_name = "olist_orders_dataset.csv"

# Título del dashboard
st.title('Predicción de ventas semanales')

# Descargar los archivos si no están presentes
if not os.path.isfile(order_items_file_name):
    order_items_filepath = download_file(order_items_url, order_items_file_name)
    st.write('Archivo de ítems descargado:', order_items_filepath)
else:
    order_items_filepath = order_items_file_name

if not os.path.isfile(orders_file_name):
    orders_filepath = download_file(orders_url, orders_file_name)
    st.write('Archivo de órdenes descargado:', orders_filepath)
else:
    orders_filepath = orders_file_name

# Cargar los datos
@st.cache_data
def load_data(items_filepath, orders_filepath):
    orders = pd.read_csv(orders_filepath)
    items = pd.read_csv(items_filepath)
    return orders, items

orders_df, items_df = load_data(order_items_filepath, orders_filepath)

# Procesamiento de los datos (Filtrado, Fusión, Agregación por Semana)
def process_data(orders_df, items_df):
    orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
    delivered_orders_df = orders_df[orders_df['order_status'] == 'delivered']
    merged_df = pd.merge(delivered_orders_df, items_df, on='order_id')
    merged_df['year_week'] = merged_df['order_purchase_timestamp'].dt.strftime('%Y-%U')
    weekly_sales = merged_df.groupby('year_week').size().reset_index(name='order_count')
    return weekly_sales

# Cargar tus datos aquí
weekly_sales = process_data(orders_df, items_df)

# Mostrar la tabla de ventas semanales
#st.write("Ventas Semanales", weekly_sales)

# Convertir year_week a un formato numérico para el modelo
weekly_sales['week_number'] = np.arange(len(weekly_sales))

# Entrenar modelo de regresión lineal
model = LinearRegression()
X = weekly_sales[['week_number']]  # Asegúrate de que X sea un DataFrame para sklearn
y = weekly_sales['order_count']
model.fit(X, y)

# Calcular métricas de exactitud para el modelo
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Mostrar métricas en Streamlit
st.write(f"Error Medio Cuadrático (MSE): {mse}")
st.write(f"Coeficiente de Determinación (R²): {r2}")

# Preparar rango de semanas para predicciones
# Asumiendo que quieres predecir desde la última semana conocida hasta 52 semanas más
last_week_num = weekly_sales['week_number'].iloc[-1]
prediction_weeks = np.arange(last_week_num + 1, last_week_num + 53).reshape(-1, 1)
predictions = model.predict(prediction_weeks)

# Preparar datos para la visualización
predicted_sales = pd.DataFrame({
    'week_number': prediction_weeks.flatten(),
    'order_count': predictions
})

# Unir los datos reales con las predicciones para la visualización
all_sales = pd.concat([weekly_sales, predicted_sales])

# Visualización con Streamlit
st.title('Número de Pedidos por Semana (2017-2019)')
fig, ax = plt.subplots(figsize=(10, 5))

# Datos reales y predicciones
ax.plot(weekly_sales['week_number'], weekly_sales['order_count'], label='Datos Reales', marker='o')
ax.plot(predicted_sales['week_number'], predicted_sales['order_count'], label='Predicciones', linestyle='--', color='orange', marker='o')

# Línea de inicio de predicciones
ax.axvline(x=last_week_num, color='red', linestyle='--', label='Inicio de las Predicciones')

# Mejorar la visualización
ax.set_xlabel('Número de Semana')
ax.set_ylabel('Número de Pedidos')
ax.legend()
ax.grid(False)

st.pyplot(fig)