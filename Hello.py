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

# Calcular el número total de semanas hasta el final de 2019
last_date_in_data = datetime.datetime.strptime(weekly_sales['year_week'].iloc[-1] + '-1', '%Y-%U-%w')
end_of_2019 = datetime.datetime.strptime('2019-12-31', '%Y-%m-%d')
total_weeks = (end_of_2019 - last_date_in_data).days // 7

# Preparar rango de fechas para predicciones
prediction_weeks = np.arange(len(weekly_sales), len(weekly_sales) + total_weeks).reshape(-1, 1)

# Verificar si hay semanas para predecir
if len(prediction_weeks) > 0:
    # Hacer predicciones
    predictions = model.predict(prediction_weeks)

# Preparar datos para la visualización
all_weeks = pd.DataFrame({
    'year_week': [f'2019-{str(week).zfill(2)}' for week in range(1, end_week_2019 - len(weekly_sales) + 1)],
    'week_number': range(len(weekly_sales), end_week_2019),
    'predicted_count': predictions
})

# Visualización
fig, ax = plt.subplots()
ax.plot(weekly_sales['year_week'], weekly_sales['order_count'], label='Datos Reales')
ax.plot(all_weeks['year_week'], all_weeks['predicted_count'], label='Predicciones', linestyle='--')
ax.set_xlabel('Semana (Desde 2017)')
ax.set_ylabel('Número de Pedidos')
ax.set_title('Número de Pedidos por Semana (2017-2019)')
ax.legend()
plt.xticks(rotation=90)
plt.tight_layout()

# Mostrar el gráfico en Streamlit
st.pyplot(fig)