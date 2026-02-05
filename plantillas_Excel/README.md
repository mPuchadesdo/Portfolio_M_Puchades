# Plantillas para Excel

¡Bienvenido/a! 
En esta carpeta del repositorio voy a compartir diferentes plantillas de Excel descargables, con una breve explicación de su uso y de lo que hay detrás de cada una.

---

## Plantilla de balance económico (balance_económico.xlsx)

Esta plantilla está compuesta de una hoja con el resumen económico del año y 12 hojas más, una para cada mes, con los ingresos y gastos detallados. 
Cuenta con:
- Una primera hoja con el resumen económico del año, que se va rellenando automáticamente conforme introducimos los datos de cada mes en su correspondiente hoja. Contiene los ingresos y gastos totales de cada mes y calcula el su balance y el acumulativo mes a mes. También tiene dos gráficas, una con el ahorro mensual y otra con los ingresos/gastos mensuales.
- He introducido diferentes valores que se calculan automáticamente:
    - Mayor, medio y mínimo ingreso
    - Mayor, medio y mínimio gasto
    - Objetivo de ahorro mensual: es un valor que puede modificarse y cambia el formato condicional: si se supera ese valor en el balance mensual, la celda se colorea de verde, si no se supera pero es positivo, de amarillo y si el balance es negativo, de rojo.
- Una hoja para cada mes, en la que encontramos 3 tablas, las cuales se pueden filtrar por todas sus columnas y una gráfica:
    - Ingresos del mes: tiene 3 columnas, concepto, categoría y monto.
    - Gastos del mes: tiene las mismas 3 columnas.
    - Gastos por categoría: es una tabla dinámica que contiene la suma de los gastos por categoría, con dos columnas, la de categoría y la de suma.
    - Gráfica de barras agrupadas asociado a la tabla dinámica de gastos por categoría, se actualiza a la vez que la tabla.

He intentado que sea una plantilla con una estética intuitiva y sencilla, que permita analizar los ingresos y gastos personales. Hay que tener en cuenta que es una plantilla enfocada en controlar la economía personal, del día a día, a nivel usuario, por ello buscamos esa simplicidad y manejabilidad.
Si quieres utilizarla, lo único que tienes que hacer es descargarla de mi portfolio, cambiar los datos introducidos en la tabla de ingresos y de gastos de cada mes por los tuyos y actualizar la tabla de gastos por categoría. De esta manera se actualizará de manera automática la hoja de Resumen anual.

---

## Plantilla de clases grupales (clases_grupales.xlsm)

Esta plantilla propone un calendario semanal sencillo para poder organizar los grupos de entrenamiento, pilates, etc. en centros pequeños, por ejemplo, de fisioterapia o de entrenamiento funcional.
Cuenta con:
- Una primera hoja a modo de horario, con una leyenda sencilla y en la que, al llenarse un grupo, los nombres pasen directamente a negrita, siendo así más visual y encontrando rápidamente los grupos con algún hueco.
- Dos hojas (una para pilates y otra para entrenamiento funcional) con una tabla cada una para anotar de manera ordenada pacientes interesadas en apuntarse a las clases grupales, con un botón asociado a una macro que genera una fila nueva en la tabla, creando un nuevo ID y anotando automáticamente la fecha del día en que se añade. Al tratarse de tablas, se puede utilizar el filtro por columnas para, por ejemplo, ver solo quién quiere un día, los lunes y a partir de las 18:00.
- Control de errores en las tablas, para evitar que se escriban números de teléfono incorrectos, datos en columnas equivocadas, etc. Es muy probable que ese Excel lo utilicen diferentes personas, en muchas ocasiones con prisa, porque mientras apuntas a la persona interesada, estás pendiente de si llega tu siguiente paciente o no.
- Una hoja oculta con listas para crear desplegables.

---

## 🛠️ Tecnologías y Herramientas

- **Herramientas**: Excel  

---

## 📫 Contacto

- 📧 Correo: [mpuchadesdo98@gmail.com](mailto:mpuchadesdo98@gmail.com)  
- 💼 LinkedIn: [Mariano Puchades](https://www.linkedin.com/in/mariano-puchades-del-olmo-325957176/)  
- 🗂️ Portfolio GitHub: [Repositorio Principal](https://github.com/mPuchadesdo/Portfolio_M_Puchades)

---

## 📄 Licencia

Este repositorio se encuentra bajo la licencia [MIT](https://opensource.org/licenses/MIT). Puedes utilizar y modificar el código según tus necesidades, siempre y cuando se otorgue el crédito correspondiente.
