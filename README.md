## 🌿 CannModel mini version español
       Recomendador de cepas de cannabis en español, puedes testearlo .
↓↓↓
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-blue?logo=huggingface)](https://huggingface.co/spaces/M4tuuc/cannmodel_es)    
---

### CannModel
    Esta version cuenta con una menor cantidad de muestras de cannabis (100) que son mas consumidas en la region de sudamerica.
*Basado en tecnicas de NPL y visualización de datos para ayudar a los usuarios a encontrar cepas de cannabis que coincidan con los efectos que buscan de una manera mas visual.
Los usuarios pueden ingresar un efecto deseado (como "relajante" o "energizante") y obtener recomendaciones personalizadas incluyendo datos como el THC y CBD tambien.*

![alt text](https://i.imgur.com/nX1TtFl.png)

---

### Arquitectura 
*Esta aplicación utiliza embeddings de palabras basado en Word2Vec para representar los efectos y sabores de las cepas de cannabis. A través de Streamlit los usuarios ingresan un efecto deseado, y se calcula la similitud entre el embedding del efecto ingresado y los embeddings precomputados de las cepas, retornando recomendaciones personalizadas. Además, incluye visualizaciones interactivas para analizar las propiedades de las cepas.*

![diagrama](https://i.imgur.com/43EDMvO.png)

---

### Requisitos
Dependencias:

streamlit

pandas

numpy

gensim

plotly

### Instalamos la dependecias
```python
pip install -r requirements.txt
```

---

 ## Arranque
 1-Clonamos el repositorio
 ```python
git clone https://github.com/m4tuuc/CannModel_mini_es.git
```
*Iniciamos la aplicacion*
```python
streamlit run app.py
```
---

Licencia

*Este proyecto está bajo la Licencia MIT.*

email: m4tuuc@gmail.com

