import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from sklearn.linear_model import LinearRegression
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
# orders_df = ...
# items_df = ...
weekly_sales = process_data(orders_df, items_df)

# Mostrar la tabla de ventas semanales
st.write("Ventas Semanales", weekly_sales)

# Convertir year_week a un formato numérico para el modelo
weekly_sales['week_number'] = np.arange(len(weekly_sales))

# Modelo de regresión lineal
model = LinearRegression()
weekly_sales['week_number'] = np.arange(len(weekly_sales))
X = weekly_sales['week_number'].values.reshape(-1, 1)
y = weekly_sales['order_count'].values
model.fit(X, y)

# Extendemos las predicciones hasta finales de 2019
# Para esto, primero necesitamos encontrar el número de la semana 52 de 2019
last_known_week = datetime.datetime.strptime('2018-35', '%Y-%W')
end_of_2019 = datetime.datetime.strptime('2019-52', '%Y-%W')
weeks_to_predict = (end_of_2019 - last_known_week).days // 7

# Creamos un nuevo DataFrame para las semanas a predecir
future_weeks = np.arange(len(weekly_sales), len(weekly_sales) + weeks_to_predict).reshape(-1, 1)
future_sales = pd.DataFrame({
    'week_number': future_weeks.flatten(),
    'year_week': [last_known_week + datetime.timedelta(weeks=w) for w in range(weeks_to_predict)],
    'order_count': model.predict(future_weeks)  # Usamos el modelo para predecir
})

# Visualización
st.title('Número de Pedidos por Semana (2017-2019)')
fig, ax = plt.subplots(figsize=(10, 5))

# Datos reales
ax.plot(weekly_sales['week_number'], weekly_sales['order_count'], label='Datos Reales', marker='o')

# Predicciones
ax.plot(future_sales['week_number'], future_sales['order_count'], label='Predicciones', linestyle='--', color='orange', marker='o')

# Línea de inicio de predicciones
start_prediction_week = weekly_sales['week_number'].iloc[-1]
ax.axvline(x=start_prediction_week, color='red', linestyle='--', label='Inicio de las Predicciones (Sept 2018)')

# Mejoramos la visualización
ax.set_xticks(weekly_sales['week_number'])
ax.set_xticklabels(weekly_sales['year_week'], rotation=90)
ax.set_xlim([0, future_sales['week_number'].iloc[-1]])
ax.set_xlabel('Semana (Desde 2017)')
ax.set_ylabel('Número de Pedidos')
ax.legend()
ax.grid(True)

st.pyplot(fig)