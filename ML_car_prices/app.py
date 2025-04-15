import streamlit as st
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

# Funciones de utilidad
def lowercase_info(x):
    return x.lower().strip()

def clasificar_potencia(power):
    if power < 100:
        return 'Baja'
    elif power < 150:
        return 'Media'
    elif power < 200:
        return 'Alta'
    else:
        return 'Muy Alta'

# Configuración inicial del proyecto
st.set_page_config(
    page_title = 'Precio de tu coche',
    page_icon = ':tractor:',
    layout = 'wide',
    initial_sidebar_state = 'expanded',
)

# Cargar modelo previamente entrenado
current_dir = os.path.dirname(__file__)
preprocessor_path = os.path.join(current_dir, "preprocessor.joblib")
preprocessing = joblib.load(preprocessor_path)
model_path = os.path.join(current_dir, "car_price_model.joblib")
modelo = joblib.load(model_path)

# Título de la app
st.title('¿Cuál es el precio de tu coche?')

# Entradas del usuario
make = st.selectbox('Marca', ['Peugeot', 'Dodge', 'Suzuki', 'Volkswagen', 'Citroen', 'Mercedes-Benz', 'Renault', 'Ferrari',
                              'Nissan', 'Maserati', 'Honda', 'Hyundai', 'Audi', 'Ford', 'BMW', 'Lancia', 'Infiniti', 
                              'Fiat', 'Jeep', 'Opel', 'Mitsubishi', 'Subaru', 'Land Rover', 'Dacia', 'Toyota', 'Volvo', 'Lexus',
                              'KIA', 'Mazda', 'Jaguar', 'Skoda', 'SEAT', 'Alfa Romeo', 'Cadillac', 'Chevrolet', 'SsangYong', 
                              'Aston Martin', 'Porsche', 'Abarth', 'MINI', 'CUPRA', 'Bentley', 'Lotus', 'Tesla', 'DS', 'Isuzu', 
                              'Tata', 'KTM', 'Lamborghini', 'Saab', 'MG', 'Chrysler', 'Daewoo', 'Iveco', 'Corvette', 'Galloper', 
                              'McLaren', 'Mahindra', 'Hummer', 'Alpine', 'Santana', 'Rover', 'Daihatsu', 'Renault Trucks',
                              'Lada', 'VAZ'])
model = st.text_input('Modelo del coche')
year = st.slider('Año de fabricación', 1960, 2025, 2012)
fuel = st.selectbox('Tipo de combustible', ['Gasolina', 'Diésel', 'Otros', 'Eléctrico'])
shift = st.selectbox('Transmisión', ['Manual', 'Automatic'])
power = st.slider('Potencia (CV)', 45, 500, 110)
cylinders_capacity = st.number_input('Cilindrada', min_value = 0.0, max_value = 6.8, value = 1.6)
emission_label = st.selectbox('Etiqueta medioambiental', ['A', 'B', 'C', 'ZERO'])
kms = st.number_input('Kilometraje', min_value = 0, max_value = 2000000, value = 100000)
dealer_zip_code = st.number_input('Código postal', min_value = 0, max_value = 60000, value = 28800)

# Botón para hacer predicción
if st.button('Predecir precio'):
    # Crear un DataFrame con los datos del usuario
    df_temp = pd.DataFrame({
        'make': [make],
        'model': [model],
        'year': [year],
        'kms': [kms],
        'fuel': [fuel],
        'shift': [shift],
        'power': [power],
        'dealer_zip_code': [dealer_zip_code],
        'emission_label': [emission_label],
        'cylinders_capacity': [cylinders_capacity]
    })

    df_temp['make'] = df_temp['make'].apply(lowercase_info)
    df_temp['model'] = df_temp['model'].apply(lowercase_info)
    df_temp['power_cat'] = df_temp['power'].apply(clasificar_potencia)
    df_temp['kms_years'] = df_temp.apply(lambda row: row['kms'] / (2024 - row['year']), axis = 1)

    cols_to_log = ['kms', 'power', 'dealer_zip_code', 'cylinders_capacity', 'kms_years']
    total_cat_feat = ['fuel', 'shift', 'make', 'model', 'power_cat', 'emission_label']
    
    entrada = preprocessing.transform(df_temp)

    prediccion = modelo.predict(entrada)  # Usa el modelo sobre los datos preprocesados
    st.success(f'El precio estimado del coche es: {prediccion[0]:,.2f} €')
    st.balloons()