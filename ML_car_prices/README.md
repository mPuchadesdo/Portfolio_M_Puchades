# El precio de los coches de segunda mano en España

## Descripción
Debido a los cambios actuales en el mercado automovilístico en España, este proyecto trata de establecer el precio de coches de segunda mano según sus características y estado de desgaste, proporcionando una guía sobre la que estimar el coste.

Los datos han sido obtenidos de DataMarket, contando con los anuncios de las principales páginas de venta de coches de segunda mano. Los datos son principalmente del año 2023, e incluyen la siguiente información:
- color: Color del vehículo.
- currency: Moneda en la que está definido el precio del vehículo.
- date: Fecha de extracción de la información.
- dealer_address: Dirección del anunciante.
- dealer_city: Ciudad del anunciante.
- dealer_country_code: Código de país del anunciante
- dealer_description: Descripción del anunciante.
- dealer_is_professional: Determina si el anunciante es o no profesional.
- dealer_name: Vendedor del vehículo. En el caso de vendedores particulares (no concesionarios), esta información está encriptada en el dataset para cumplir con la GDPR.
- dealer_registered_at: Fecha de registro del anunciante en la plataforma.
- dealer_website: Página web del anunciante.
- dealer_zip_code: Código postal del anunciante.
- description: Descripción presente en el anuncio del vehículo.
- doors: Número de puertas del vehículo.
- fuel: Tipo de combustible del vehículo (diésel, gasolina, eléctrico, híbrido).
- is_professional: Indica si el vendedor es profesional (un concesionario).
- kms: Kilometraje del vehículo.
- location: Ciudad en la que se ha publicado el anuncio.
- make: Marca del vehículo.
- model: Modelo del vehículo.
- photos: Número de fotos del vehículo disponibles en el anuncio.
- power: Potencia del vehículo.
- price: Precio de venta del vehículo.
- publish_date: Fecha de publicación del anuncio.
- shift: Tipo de cambio (Automático/Manual).
- update_date: Fecha de actualización del anuncio.
- vehicle_type: Tipo de vehículo: coche, moto...
- version: Versión del vehículo.
- year: Año de fabricación del vehículo.

Pueden acceder a parte de los datos solicitándolos en la web de [DataMarket](https://datamarket.es).

## Estructura del Proyecto
```
/
|-- src/data_sample/                 # Contiene una muestra del dataset utilizado
|-- src/modeling_process/        
|   |-- modeling_process_ES.ipynb    # Notebook que explica el proceso de preparado de las variables y entrena el modelo definitivo (en castellano)
|   |-- modeling_process_ENG.ipynb   # Notebook que explica el proceso de preparado de las variables y entrena el modelo definitivo (en inglés)
|-- src/utils/                       # Contiene archivos de utilidad (funciones de visualización, etc.)
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
git clone https://github.com/tu_usuario/tu_repositorio.git
cd tu_repositorio
git lfs install
git lfs pull
```

2. Ejecutar el Jupyter Notebook para el entrenamiento:

Abrir `results_notebook/results_notebook.ipynb` y ejecutar para ver el proceso de modelado.


## Modelo
Se ha implementado un modelo de regresión **RandomForestRegressor** de `scikit-learn`, ya que proporcionó los mejores resultados en pruebas, aunque se sigue intentando mejorar el resultado de sus predicciones.
Las principales características del modelo son:
- Ha sido entrenado con un dataset de coches de segunda mano de 38 mil instancias, con atributos como: marca, modelo, año, kilometraje, combustible, cambio, etc.
- Se han ajustado hiperparámetros mediante `RandomizedSearch` para mejorar el rendimiento.

________________________________________________________________________________________________________________________________________________________________________________________________________________________

# Used cars prices in Spain

## Description
Due to the current changes in the automotive market in Spain, this project aims to establish the price of used cars based on their characteristics and wear condition, providing a guide for estimating their cost.

The data has been obtained from DataMarket, containing listings from the main second-hand car sales websites. The data is primarily from the year 2023 and includes the following information:
- color: Vehicle color.
- currency: Currency in which the vehicle price is defined.
- date: Date of data extraction.
- dealer_address: Advertiser's address.
- dealer_city: Advertiser's city.
- dealer_country_code: Advertiser's country code.
- dealer_description: Advertiser's description.
- dealer_is_professional: Indicates whether the advertiser is a professional or not.
- dealer_name: Vehicle seller. For private sellers (not dealerships), this information is encrypted in the dataset to comply with GDPR.
- dealer_registered_at: Date the advertiser registered on the platform.
- dealer_website: Advertiser's website.
- dealer_zip_code: Advertiser's postal code.
- description: Vehicle advertisement description.
- doors: Number of vehicle doors.
- fuel: Vehicle fuel type (diesel, gasoline, electric, hybrid).
- is_professional: Indicates whether the seller is a professional (a dealership).
- kms: Vehicle mileage.
- location: City where the advertisement was published.
- make: Vehicle brand.
- model: Vehicle model.
- photos: Number of vehicle photos available in the ad.
- power: Vehicle power.
- price: Vehicle selling price.
- publish_date: Date the ad was published.
- shift: Transmission type (Automatic/Manual).
- update_date: Date the ad was updated.
- vehicle_type: Type of vehicle: car, motorcycle, etc.
- version: Vehicle version.
- year: Vehicle manufacturing year.

Part of the data can be accessed by requesting it from the [DataMarket](https://datamarket.es) website.

## Project Structure
```
/
|-- src/data_sample/                 # Contains a sample of the dataset
|-- src/modeling_process/        
|   |-- results_notebook_ENG.ipynb   # Notebook explaining the variable preparation process and training the final model (English version)
|   |-- results_notebook_ES.ipynb    # Notebook explaining the variable preparation process and training the final model (Spanish version)
|-- src/utils/                       # Contains utility files (visualization functions, etc.)
|-- requirements.txt                 # Libraries required for the project
|-- README.md                        # Documentation
```

## Installation
To run the project, make sure you have Python (>=3.9) installed along with the following libraries:

```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:

```bash
git clone https://github.com/your_user/your_repository.git
cd your_repository
git lfs install
git lfs pull
```

2. Run the Jupyter Notebook for training:

Open `results_notebook/results_notebook.ipynb` and execute it to see the modeling process.

## Model
A **RandomForestRegressor** model from `scikit-learn` has been implemented, as it provided the best results in tests, although improvements are still being explored to enhance prediction accuracy.

The main characteristics of the model are:
- It has been trained on a second-hand car dataset with 38,000 instances, including attributes such as brand, model, year, mileage, fuel type, transmission, etc.
- Hyperparameters have been fine-tuned using `RandomizedSearch` to improve performance.
