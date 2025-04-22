
# El precio de los coches de segunda mano en España
[Click aquí para probar el modelo en una aplicación web](https://huggingface.co/spaces/mpuchdo/El_precio_de_su_coche)

## Descripción
Debido a los cambios actuales en el mercado automovilístico en España, este proyecto trata de establecer el precio de coches de segunda mano según sus características y estado de desgaste, proporcionando una guía sobre la que estimar el coste.

Los datos han sido obtenidos de DataMarket, contando con los anuncios de las principales páginas de venta de coches de segunda mano. Los datos son principalmente del año 2023, e incluyen la siguiente información:
- dealer_zip_code: Código postal del anunciante.
- fuel: Tipo de combustible del vehículo (diésel, gasolina, eléctrico, híbrido).
- kms: Kilometraje del vehículo.
- make: Marca del vehículo.
- model: Modelo del vehículo.
- power: Potencia del vehículo.
- price: Precio de venta del vehículo.
- shift: Tipo de cambio (Automático/Manual).
- version: Versión del vehículo.
- year: Año de fabricación del vehículo.
El dataset original tiene un mayor número de columnas, pero fueron descartadas porque no aportaban valor predictivo al modelo. 
Pueden acceder a parte de los datos solicitándolos en la web de [DataMarket](https://datamarket.es).

## Estructura del Proyecto
```
/
|-- src/
|-- src/data_sample/                 # Contiene una muestra del dataset utilizado
|-- src/modeling_process/        
|   |-- modeling_process_ES.ipynb    # Notebook que explica el proceso de preparado de las variables y entrena el modelo definitivo (en castellano)
|   |-- modeling_process_ENG.ipynb   # Notebook que explica el proceso de preparado de las variables y entrena el modelo definitivo (en inglés)
|-- src/utils/                       # Contiene archivos de utilidad (funciones de visualización, etc.)
|-- car_price_model.joblib           # Modelo entrenado resultante del proyecto
|-- preprocessor.joblib              # Preprocesador de variables para que la aplicación web funcione
|-- gradio_app.py                    # Aplicación web
|-- requirements.txt                 # Librerías necesarias para el proyecto
|-- README.md                        # Documentación
```

## Instalación
Para ejecutar el proyecto, asegúrese de tener instalado Python (>=3.9) y las siguientes librerías:

```bash
pip install -r requirements.txt
```

## Uso
1. Clonar el repositorio:

```bash
git clone https://github.com/mPuchadesdo/Portfolio_M_Puchades/main/ML_car_prices/
cd <tu_repositorio>
```

2. Ejecutar el Jupyter Notebook para el entrenamiento:

Abrir `modeling_process/modeling_process_ES.ipynb` y ejecutar para ver el proceso de modelado.


## Modelo
Se ha implementado un modelo de regresión **RandomForestRegressor** de `scikit-learn`, ya que proporcionó los mejores resultados en pruebas, aunque se sigue intentando mejorar el resultado de sus predicciones.
Las principales características del modelo son:
- Ha sido entrenado con un dataset de coches de segunda mano de 38 mil instancias, con atributos como: marca, modelo, año, kilometraje, combustible, cambio, etc.
- Se han ajustado hiperparámetros mediante `RandomizedSearch` para mejorar el rendimiento.
