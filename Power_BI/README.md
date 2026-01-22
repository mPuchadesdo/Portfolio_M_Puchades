# 📊 Proyectos con Power BI

¡Bienvenido/a! 
En esta carpeta del repositorio voy a compartir diferentes proyectos realizados con Power BI y otras herramientas relacionadas.

---

## Estudios de fisioterapia

### [Evaluación del trapecio inferior con modo M](https://github.com/mPuchadesdo/Portfolio_M_Puchades/blob/main/Power_BI/Estudios_fisioterapia/Evaluaci%C3%B3n_trapecio_inferior/EVALUACI%C3%93N%20DEL%20TRAPECIO%20INFERIOR%20CON%20MODO%20M%20EN%20DOLOR%20DE%20CUELLO.pdf)

Objetivo: estudiar si existen diferencias en la función del trapecio inferior entre sujetos sanos y sujetos con dolor cervical unilateral por medio de distintas variables ecográficas obtenidas mediante modo M.

Métodos: se condujo un estudio de casos y controles en el que participaron 10 sujetos con dolor cervical unilateral y 10 controles. 4 variables ecográficas del TI (grosor en reposo, grosor en contracción, velocidad de activación, velocidad de relajación) fueron recogidas durante una contracción isométrica máxima voluntaria (MCIV) de 5 segundos, evaluada mediante dinamometría. Las mediciones se realizaron en ambos lados. 

Resultados: no se observaron diferencias significativas entre los sujetos con y sin dolor de cuello para ninguna de las variables evaluadas (p>0,05). A pesar de esto, se pudo observar una tendencia de mayor velocidad de activación del TI en los sujetos sanos respecto de los sujetos con dolor cervical unilateral siendo esta más evidente en el lado dominante (0,41±0,18 vs 0,29±0,13). 

Conclusiones: los resultados de este estudio no muestran relación entre las variables medidas y el dolor de cuello. Sin embargo, por la tendencia que muestra la velocidad de activación del TI en sujetos sanos en comparación a los sujetos con dolor, podría ser una variable para tener en cuenta en futuros estudios de investigación respecto al dolor de cuello.

📊 Desarrollo en Power BI

Para acceder al informe creado con Power BI, [pulse aquí](https://app.powerbi.com/links/inpruUXyne?ctid=ced2c552-7d1f-4731-aa3a-2f0ec9629e26&pbi_source=linkShare&bookmarkGuid=c882cfdb-24ba-4c60-89e3-32e2fb88e19d).

Los datos estaban recogidos en un Excel, por lo que tuve que limpiarlos y hacer lo siguiente con ellos:
- Realicé un miniproceso de ETL para poder utilizar los datos: mediante Power Query cambié el nombre de columnas, los tipos de datos y sustituí caracteres como "0" y "1" por "control" y "dolor", por ejemplo. 
- Creé medidas para obtener datos como la Edad media, el número de sujetos pertenecientes al grupo control y al grupo de casos, etc. Esto permite respetar el contexto de filtro y conseguir que sean dinámicas.
- Diseñé un pequeño borrador imaginando cómo me gustaría a mí que me mostraran los datos del estudio, de forma que pudiera ser fácilmente entendible y que contara los resultados principales y posibles interpretaciones. 
- Trabajé con la parte estética, intentando que quedara un informe limpio y atractivo.

⚠️ Limitaciones

- Tamaño muestral reducido
- Estudio piloto exploratorio

---

## 🛠️ Tecnologías y Herramientas

- **Herramientas**: Power BI · Excel · DAX

---

## 📫 Contacto

- 📧 Correo: [mpuchadesdo98@gmail.com](mailto:mpuchadesdo98@gmail.com)  
- 💼 LinkedIn: [Mariano Puchades](https://www.linkedin.com/in/mariano-puchades-del-olmo-325957176/)  
- 🗂️ Portfolio GitHub: [Repositorio Principal](https://github.com/mPuchadesdo/Portfolio_M_Puchades)

---

## 📄 Licencia

Este repositorio se encuentra bajo la licencia [MIT](https://opensource.org/licenses/MIT). Puedes utilizar y modificar el código según tus necesidades, siempre y cuando se otorgue el crédito correspondiente.
