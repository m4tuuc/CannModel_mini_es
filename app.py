import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data  
def load_data():
    df = pd.read_csv("cepas.csv")
    df["THC"] = df["THC"].astype(str)
    
    def convert_thc(val):
        val = val.strip().lower()
        if not val or val == 'nan':
            return 0.0
        if '%' in val:
            val = val.replace('%', '').strip()
        try:
            return float(val) / 100
        except ValueError:
            thc_map = {"bajo": 0.15, "medium": 0.20, "alto": 0.25}
            return thc_map.get(val, 0.18)
    
    df["THC"] = df["THC"].apply(convert_thc)
    
    df["cbd_texto"] = df["CBD"].astype(str).str.strip()
    df["CBD"] = df["CBD"].astype(str).str.strip().str.lower()
    cbd_map = {"bajo": 0.05, "medium": 0.1, "alto": 0.2}
    df["CBD"] = df["CBD"].map(cbd_map).fillna(0.05)
    
    if "train" not in df.columns:
        df["train"] = df["Efecto"] + " " + df["Sabor"]
    return df

@st.cache_resource
def load_model():
    try:
        from gensim.models import KeyedVectors
        return KeyedVectors.load("model_gensim.kv")
    except:
        df = load_data()
        texts = [text.lower().split() for text in df["train"]]
        model = Word2Vec(texts, vector_size=150, window=5, min_count=1, workers=4)
        model.wv.save("model_gensim.kv")
        return model.wv


df = load_data()
model = load_model()

# Generar embeddings para las cepas
df["embedding"] = df["train"].apply(
    lambda x: np.mean([model[word] for word in x.lower().split() if word in model], axis=0)
    if any(word in model for word in x.lower().split())
    else np.zeros(150)
)

def get_cepas_similares(efecto_usuario, n=3, top_n=10):
    palabras = efecto_usuario.lower().split()
    palabras_validas = [palabra for palabra in palabras if palabra in model]
    if not palabras_validas:
        return df.sample(n)
    embedding_usuario = np.mean([model[palabra] for palabra in palabras_validas], axis=0)
    similitudes = df["embedding"].apply(
        lambda emb: np.dot(emb, embedding_usuario) / 
        (np.linalg.norm(emb) * np.linalg.norm(embedding_usuario))
        if np.linalg.norm(emb) > 0 else 0
    )
    df["similitud"] = similitudes
    top_similar = df.sort_values("similitud", ascending=False).head(top_n)
    return top_similar.sample(min(n, len(top_similar)))

# Interfaz de Streamlit
st.title("🌿 Recomendador de cepas de cannabis")
st.subheader("Recomendaciones basadas en efectos deseados")

# Inicializar el estado si no existe
if 'cepas' not in st.session_state:
    st.session_state.cepas = df.sample(3)
if 'efecto_anterior' not in st.session_state:
    st.session_state.efecto_anterior = ""
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "tab1"
if 'selected_strain' not in st.session_state:
    st.session_state.selected_strain = None

# Sidebar para entrada de efecto y boton de refresh.
seleccion = st.sidebar.selectbox('Cepa (100 muestras)', options=df['Cepa'], index=0)

if st.sidebar.button("Seleccionar"):
    st.session_state.selected_strain = seleccion
    st.session_state.active_tab = "tab3"

efecto_buscado = st.sidebar.text_input("🔍 Escribe el efecto deseado", "")
refresh = st.sidebar.button("🔄 Actualizar cepas")


efecto_cambio = efecto_buscado != st.session_state.efecto_anterior

# Actualizar cepas SOLO si el efecto cambia o se presiona Refresh
if efecto_cambio or refresh:
    if efecto_buscado:
        st.session_state.cepas = get_cepas_similares(efecto_buscado)
    else:
        st.session_state.cepas = df.sample(3)
    st.session_state.efecto_anterior = efecto_buscado

#pestañas
tab1, tab2, tab3 = st.tabs(["Recomendaciones", "Gráficos y Comparativas", "General"])

# Pestaña 1: Recomendaciones y detalles
with tab1:
    # Mostrar las 3 cepas recomendadas
    st.write("Cepas recomendadas:")
    cols = st.columns(3)
    for i, (_, row) in enumerate(st.session_state.cepas.iterrows()):
        with cols[i]:
            # Truncar nombre si es muy largo
            nombre = row["Cepa"][:20] + "..." if len(row["Cepa"]) > 20 else row["Cepa"]
            st.subheader(nombre)
            if st.button("Seleccionar", key=f"btn_{i}"):
                st.session_state.cepa_seleccionada = row
                st.session_state.selected_strain = row["Cepa"]
                st.session_state.active_tab = "tab3"
                st.rerun()

    # Mostrar detalles de la cepa seleccionada
    if "cepa_seleccionada" in st.session_state:
        cepa = st.session_state.cepa_seleccionada
        st.divider()
        st.header(cepa["Cepa"])
        
        st.subheader("Composición")
        st.write(f"**THC:** {cepa['THC']*100:.1f}%")
        st.write(f"**CBD:** {cepa['cbd_texto']}")
        st.write(f"**Origen:** {cepa.get('Origen génetico', 'No disponible')}")
        
        thc_data = pd.DataFrame({
            'Componente': ['THC'],
            'Porcentaje': [cepa['THC']*100]
        })
        fig = px.bar(thc_data, x='Componente', y='Porcentaje', 
                     title="Nivel de THC", 
                     labels={'Porcentaje': 'Porcentaje (%)'},
                     color_discrete_sequence=['#76b852'])
        fig.update_layout(yaxis_range=[0, 30])
        st.plotly_chart(fig)

        st.subheader("Características")
        col1, col2 = st.columns(2)
        with col1:
            st.write("🚀**Efectos:**")
            for efecto in cepa["Efecto"].split(", "):
                st.write(f"- {efecto}")
        with col2:
            st.write("👅**Sabores:**")
            for sabor in cepa["Sabor"].split(", "):
                st.write(f"- {sabor}")
        
        # Botón para volver a cepas aleatorias
        if st.button("Volver a cepas aleatorias"):
            del st.session_state.cepa_seleccionada
            st.session_state.cepas = df.sample(3)

with tab2:
    st.header("Gráficos y Comparativas")

    df_exploded = df.assign(Sabor=df['Sabor'].str.split(', ')).explode('Sabor')
    df_exploded['Sabor'] = df_exploded['Sabor'].str.strip().str.lower()

    conteo = df_exploded['Sabor'].value_counts().nlargest(5)

    fig = px.pie(
        names=conteo.index,
        values=conteo.values,
        title='Predominancia de sabores',
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    # Personalizar etiquetas y diseño
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        rotation=45,
        pull=[0.1 if i == 0 else 0 for i in range(len(conteo))]  # Destacar el predominante
    )

    fig.update_layout(
        uniformtext_minsize=13,
        uniformtext_mode='hide',
        title_x=0.6,
        showlegend=True
    )

    st.plotly_chart(fig)
    
    df_sorted = df.sort_values(by='Efecto')
    df_sorted['rank'] = df_sorted.groupby('Cepa').cumcount()
  
    df_limit = df_sorted[df_sorted['rank'] < 2].copy()
    top5 = df_limit['Efecto'].value_counts().nlargest(5).index
    df_limit = df_limit[df_limit['Efecto'].isin(top5)]
    
    efectos_validos = df_limit.groupby('Efecto')['Cepa'].nunique()
    efectos_validos = efectos_validos[efectos_validos >= 2].index
    df_limited = df_limit[df_limit['Efecto'].isin(efectos_validos)]

    fig = px.sunburst(
        df_limited,
        path=['Efecto', 'Cepa'],
        title="Efectos y Cepas"
    )
    
    fig.update_traces(textfont=dict(size=13))
    fig.update_layout(
        width=650,   # Ajusta el ancho
        height=650,  # Ajusta el alto
    )
    
    st.plotly_chart(fig)


with tab3:

    if st.session_state.selected_strain:
        st.header(f"Análisis Detallado: {st.session_state.selected_strain}")
    
        strain_data = df[df['Cepa'] == st.session_state.selected_strain].iloc[0]
        

        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Características Principales")
            # Convertir THC de decimal a porcentaje para mostrar
            thc_percent = strain_data['THC'] * 100 if strain_data['THC'] < 1 else strain_data['THC']
            st.metric("THC", f"{thc_percent:.1f}%")
            
            # Usar CBD original si está disponible, sino usar el valor numérico
            if 'cbd_texto' in strain_data and strain_data['cbd_texto'] and strain_data['cbd_texto'].lower() != 'nan':
                cbd_display = strain_data['cbd_texto']
            else:
                cbd_percent = strain_data['CBD'] * 100 if strain_data['CBD'] < 1 else strain_data['CBD']
                cbd_display = f"{cbd_percent:.1f}%"
            
            st.metric("CBD", cbd_display)
            
            if 'Tipo' in strain_data:
                st.write(f"**Tipo:** {strain_data['Tipo']}")
            
            st.write(f"**Sabor Principal:** {strain_data['Sabor'].split(',')[0] if ',' in strain_data['Sabor'] else strain_data['Sabor']}")
            st.write(f"**Efecto Dominante:** {strain_data['Efecto'].split(',')[0] if ',' in strain_data['Efecto'] else strain_data['Efecto']}")
            
            if 'Origen génetico' in strain_data:
                st.write(f"**Origen genético:** {strain_data['Origen génetico']}")
            
        with col2:
            st.subheader("Perfil Cannabinoide")
            # Asegurar que THC y CBD estén en porcentaje para la gráfica
            thc_value = strain_data['THC'] * 100 if strain_data['THC'] < 1 else strain_data['THC']
            cbd_value = strain_data['CBD'] * 100 if strain_data['CBD'] < 1 else strain_data['CBD']
            
            fig = px.bar(
                x=['THC', 'CBD'],
                y=[thc_value, cbd_value],
                labels={'x': 'Cannabinoide', 'y': 'Porcentaje (%)'},
                color=['THC', 'CBD'],
                color_discrete_sequence=['#4CAF50', '#2196F3']
            )
            fig.update_layout(yaxis_range=[0, max(thc_value, cbd_value) * 1.2])
            st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar todos los efectos y sabores
        st.subheader("Efectos y Sabores Completos")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("🚀**Efectos:**")
            for efecto in strain_data["Efecto"].split(", "):
                st.write(f"- {efecto}")
        
        with col2:
            st.write("👅**Sabores:**")
            for sabor in strain_data["Sabor"].split(", "):
                st.write(f"- {sabor}")
        
        # Boton para volver
        if st.button("← Volver a Recomendaciones"):
            st.session_state.active_tab = "tab1"
            st.experimental_rerun()
    else:
        st.info("Selecciona una cepa usando el botón 'Seleccionar' en la barra lateral o en las recomendaciones para ver su información detallada.")

# Activar la pestaña seleccionada
if st.session_state.active_tab == "tab3":
    try:
        # Este script se ejecutará en el navegador para activar la pestaña 3
        js = f"""
        <script>
            window.addEventListener('load', function() {{
                setTimeout(function() {{
                    const tabs = window.parent.document.querySelectorAll('.st-tabs button[role="tab"]');
                    if (tabs.length >= 3) {{
                        tabs[2].click();
                    }}
                }}, 100);
            }});
        </script>
        """
        st.components.v1.html(js, height=0)
    except:
        pass
