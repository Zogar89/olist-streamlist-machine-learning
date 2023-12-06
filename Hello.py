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
X = weekly_sales['week_number'].values.reshape(-1, 1)
y = weekly_sales['order_count'].values
model.fit(X, y)

# Hacer predicciones
start_week = datetime.datetime.strptime('2018-09-01', '%Y-%m-%d').isocalendar()[1]
end_week = datetime.datetime.strptime('2019-12-31', '%Y-%m-%d').isocalendar()[1]

# Asegurarse de que las semanas de inicio y fin estén en el DataFrame
start_week_str = '2018-{}'.format(str(start_week).zfill(2))
end_week_str = '2019-{}'.format(str(end_week).zfill(2))

if start_week_str in weekly_sales['year_week'].values and end_week_str in weekly_sales['year_week'].values:
    start_week_number = weekly_sales[weekly_sales['year_week'] == start_week_str].week_number.values[0]
    end_week_number = weekly_sales[weekly_sales['year_week'] == end_week_str].week_number.values[0]
    prediction_weeks = np.arange(start_week_number, end_week_number + 1).reshape(-1, 1)
    predictions = model.predict(prediction_weeks)

    # Visualización
    fig, ax = plt.subplots()
    ax.plot(weekly_sales['year_week'], weekly_sales['order_count'], label='Datos Reales')
    ax.plot(weekly_sales.loc[start_week_number:end_week_number, 'year_week'], predictions, label='Predicciones', linestyle='--')
    ax.set_xlabel('Semana (Desde 2017)')
    ax.set_ylabel('Número de Pedidos')
    ax.set_title('Número de Pedidos por Semana (2017-2018)')
    ax.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)
else:
    st.error('Rango de fechas seleccionado no disponible en los datos.')
