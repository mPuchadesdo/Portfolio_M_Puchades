
import gradio as gr
import joblib
import pandas as pd
import numpy as np
import os

# Funciones auxiliares
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

# Cargar modelo y preprocesador
current_dir = os.path.dirname(__file__)
preprocessor_path = os.path.join(current_dir, "preprocessor.joblib")
preprocessing = joblib.load(preprocessor_path)
model_path = os.path.join(current_dir, "car_price_model.joblib")
modelo = joblib.load(model_path)

# Función de predicción
def predecir_precio(make, model_name, year, fuel, shift, power, cylinders_capacity, emission_label, kms, dealer_zip_code):
    df_temp = pd.DataFrame({
        'make': [make],
        'model': [model_name],
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
    df_temp['kms_years'] = df_temp.apply(lambda row: row['kms'] / (2024 - row['year']), axis=1)

    entrada = preprocessing.transform(df_temp)
    prediccion = modelo.predict(entrada)
    return f"El precio estimado de tu coche es: {prediccion[0]:,.2f} €"

# Lista de marcas (abreviada por simplicidad)
marcas = ['Peugeot', 'Dodge', 'Suzuki', 'Volkswagen', 'Citroen', 'Mercedes-Benz', 'Renault', 'Ferrari',
        'Nissan', 'Maserati', 'Honda', 'Hyundai', 'Audi', 'Ford', 'BMW', 'Lancia', 'Infiniti', 
        'Fiat', 'Jeep', 'Opel', 'Mitsubishi', 'Subaru', 'Land Rover', 'Dacia', 'Toyota', 'Volvo', 'Lexus',
        'KIA', 'Mazda', 'Jaguar', 'Skoda', 'SEAT', 'Alfa Romeo', 'Cadillac', 'Chevrolet', 'SsangYong', 
        'Aston Martin', 'Porsche', 'Abarth', 'MINI', 'CUPRA', 'Bentley', 'Lotus', 'Tesla', 'DS', 'Isuzu', 
        'Tata', 'KTM', 'Lamborghini', 'Saab', 'MG', 'Chrysler', 'Daewoo', 'Iveco', 'Corvette', 'Galloper', 
        'McLaren', 'Mahindra', 'Hummer', 'Alpine', 'Santana', 'Rover', 'Daihatsu', 'Renault Trucks',
        'Lada', 'VAZ']

demo = gr.Interface(
    fn = predecir_precio,
    inputs = [
        gr.Dropdown(choices=marcas, label = "Marca"),
        gr.Textbox(label = "Modelo del coche"),
        gr.Slider(minimum = 1960, maximum = 2025, value = 2012, step = 1, label = "Año de fabricación"),
        gr.Radio(choices = ["Gasolina", "Diésel", "Otros", "Eléctrico"], label = "Tipo de combustible"),
        gr.Radio(choices = ["Manual", "Automatic"], label="Transmisión"),
        gr.Number(minimum = 45, maximum = 500, value = 110, label = "Potencia (CV)"),
        gr.Number(minimum = 0.0, maximum = 6.8, value = 1.6, step = 0.1, label = "Cilindrada"),
        gr.Radio(choices = ["A", "B", "C", "ZERO"], label = "Etiqueta medioambiental"),
        gr.Number(label = "Kilometraje"),
        gr.Number(label = "Código postal")
    ],
    outputs = "text",
    title = "¿Cuál es el precio de tu coche?",
    description = '''
    Debido a los cambios actuales en el mercado automovilístico en España, este proyecto trata de establecer 
    el precio de coches de segunda mano según sus características y estado de desgaste, proporcionando una guía 
    sobre la que estimar el coste.
    Introduzca las características de su coche para estimar su precio.
    ''',
    article = '''
    El modelo detrás de esta aplicación está entrenado con anuncios reales, principalmente del 2023 en España. Se obtuvo 
    un RMSE contra test de 744,64€. 
    Sin embargo, hay que tener en cuenta que los valores estimados no tienen en cuenta el mercado actual.
    El proceso de creación explicado puede encontrarse en el siguiente repositorio:
    "https://github.com/mPuchadesdo/Portfolio_M_Puchades/tree/main/ML_car_prices"
    ''', 
    theme = 'soft',
)

if __name__ == "__main__":
    demo.launch(share = True)
