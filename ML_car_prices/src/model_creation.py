import sys

sys.path.insert(0, '../')
import gdown

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
import seaborn as sns
sns.set_theme(style = "whitegrid", palette = "muted", context = "notebook")
import re

from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Carga de datos
url = "https://drive.google.com/uc?id=1NtUt42nZ766HduQfyAxls1pX6deqtVh2"
output = "data.csv"
gdown.download(url, output, quiet = False)
df = pd.read_csv("data.csv")

# Target y listas de columnas de interés
target = 'price'
cols_to_drop = ['doors', 'color', 'description', 'vehicle_type', 'currency', 'dealer_name', 'dealer_description', \
                'dealer_address', 'dealer_city', 'dealer_country_code', 'dealer_is_professional', 'dealer_website', \
                'dealer_registered_at', 'date', 'publish_date', 'update_date', 'location', 'photos']
num_feat = ['year', 'kms', 'power', 'dealer_zip_code', 'kms_years', 'cylinders_capacity']
cat_feat = ['fuel', 'shift', 'make', 'model', 'power_cat', 'emission_label']
cols_to_log = ['kms', 'power', 'dealer_zip_code', 'cylinders_capacity', 'kms_years']
print('El dataset se ha cargado correctamente')

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Funciones a utilizar
## Obtención de información de la columna 'version'
def extract_info(x):
    if not isinstance(x, str):
        return pd.Series([0, 0, 0])
    
    pattern = r'(?P<number>\b\d\.\d\b)[dD]?|(?P<kw>\d+)\s*kW|(?P<cv>\d+)\s*CV'
    matches = re.findall(pattern, x, re.IGNORECASE)

    cylinders_capacity = kw = cv = 0
    for match in matches:
        if match[0]:
            cylinders_capacity = float(match[0])
        if match[1]:
            kw = float(match[1])
        if match[2]:
            cv = float(match[2])
    
    return pd.Series([cylinders_capacity, kw, cv])

## Transformación en minúsculas de todo el string
def lowercase_info(x):
    return x.lower().strip()

## Clasificador según la potencia del vehículo
def clasificar_potencia(power):
    if power < 90:
        return 'Baja'
    elif power < 150:
        return 'Media'
    elif power < 200:
        return 'Alta'
    else:
        return 'Muy Alta'

# Modas y medias para rellenar nulos
kms_mean = df['kms'].mean().round(0)
shift_mode = df['shift'].mode()[0]
dealer_zip_code_mode = df['dealer_zip_code'].mode()[0]
fuel_mode = df['fuel'].mode()[0]

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Transformaciones de las variables
## Eliminamos las columnas que no nos interesan:
df.drop(columns = cols_to_drop, inplace = True)

## Eliminamos aquellos vehículos que tengan un valor menor de 2000, ya que serán despieces o anuncios falsos:
df.drop(index = (df.loc[df[target] < 2000].index), inplace = True)

## Eliminamos los duplicados:
df.drop_duplicates(keep = 'last', inplace = True)

## Eliminamos aquellas columnas que tengan una potencia superiora 1020, puesto que son un error:
df.drop(df.loc[df.power > 1020].index, inplace = True)

## Rellenamos los nulos con la media y modas que hemos utilizado de df:
df.loc[df['kms'].isna(), 'kms'] = kms_mean
df.loc[df['shift'].isna(), 'shift'] = shift_mode
df.loc[df['dealer_zip_code'].isna(), 'dealer_zip_code'] = dealer_zip_code_mode

## Obtenemos toda la información posible de la columna de version y limpiamos make y model:
df[['cylinders_capacity', 'kW', 'CV']] = df['version'].apply(extract_info)
df['make'] = df['make'].apply(lowercase_info)
df['model'] = df['model'].apply(lowercase_info)

## Rellenamos los faltantes de power que podemos utilizando la información obtenida de version:
df.loc[df['power'].isna(), 'power'] = df.loc[df['power'].isna(), 'CV']
df.loc[df['power'] == 0, 'power'] = df.loc[df['power'] == 0, 'kW']
power_mean = df['power'].mean().round(0)
df.loc[df['power'] == 0, 'power'] = power_mean

## Rellenamos los nulos con la moda de fuel de df:
df.loc[df['fuel'].isna(), 'fuel'] = fuel_mode

## Eliminamos kW y CV:
df.drop(columns = ['kW', 'CV'], inplace = True)

## Creamos las variables de power_cat y kms_years:
actual_year = 2024
df['power_cat'] = df['power'].apply(clasificar_potencia)
df['kms_years'] = df.apply(lambda row: row['kms'] / (actual_year - row['year']), axis = 1)

## Creamos la variable de emission_label:
filter_A_test = ((df['year'] < 2001) & (df['fuel'] == 'Gasolina')) | ((df['year'] < 2006) & (df['fuel'] == 'Diésel'))
filter_B_test = ((df['year'] >= 2001) & (df['year'] < 2006) & (df['fuel'] == 'Gasolina')) | ((df['year'] >= 2006) & (df['year'] < 2015) & (df['fuel'] == 'Diésel'))
filter_C_test = ((df['year'] >= 2006) & (df['fuel'] == 'Gasolina')) | ((df['year'] >= 2015) & (df['fuel'] == 'Diésel'))
filter_ZERO_test = (df['fuel'] == 'Eléctrico')
filter_otros_test = df['fuel'] == 'Otros'
df.loc[filter_A_test, 'emission_label'] = 'A'
df.loc[filter_B_test, 'emission_label'] = 'B'
df.loc[filter_C_test, 'emission_label'] = 'C'
df.loc[filter_ZERO_test, 'emission_label'] = 'ZERO'
df.loc[filter_otros_test, 'emission_label'] = df['emission_label'].mode()[0]
print('Se han rellenado nulos y creado nuevas variables')

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

# División en X e y
X = df.drop(columns = [target, 'version'])
y = df[target]

# Preprocesado
log_transformer = FunctionTransformer(func = np.log1p)
num_pipeline = Pipeline([
    ("log", log_transformer)])

cat_pipeline = Pipeline([
    ("ohencoder", OneHotEncoder(handle_unknown = "ignore"))])

preprocessing = ColumnTransformer([
        ("cat", cat_pipeline, cat_feat),
        ("log", num_pipeline, cols_to_log)],
        remainder = "passthrough")

preprocessing.fit(X)
joblib.dump(preprocessing, '../preprocessor.joblib', compress = True)
print('El preprocesador ha sido guardado correctamente')

X = preprocessing.transform(X)
print('Los datos han sido procesados para el entrenamiento')

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Entrenamiento del modelo
rf_reg = RandomForestRegressor(n_estimators = 200,
                               min_samples_split = 2,
                               min_samples_leaf = 1,
                               max_features = 'sqrt',
                               max_depth = None,
                               bootstrap = False,
                               random_state = 42)

rf_reg.fit(X, y)
joblib.dump(rf_reg, '../car_price_model.joblib', compress = True)
print('El modelo ha sido guardado correctamente')