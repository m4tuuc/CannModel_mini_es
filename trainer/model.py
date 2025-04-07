from gensim.models import Word2Vec
import pandas as pd
import re
import unicodedata


df = pd.read_csv('cepas.csv')

# Combinar efectos y sabores para entrenar
df['train'] = df['Efecto'] + ' ' + df['Sabor']

def preprocesar_texto(texto):
    # Eliminar caracteres especiales
    texto = re.sub(r"[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]", "", texto)
    # Normalizar acentos
    texto = unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("utf-8")
    return texto

df["texto_limpio"] = df["train"].apply(preprocesar_texto)


df['train'].to_csv('train.txt', index=False, header=False, encoding="utf-8")
print("Archivo de entrenamiento creado con éxito")


sentences = [text.lower().split() for text in df['texto_limpio']]

# Entrenar modelo Word2Vec
model = Word2Vec(
    sentences, 
    vector_size=150, 
    window=5,        
    min_count=1,     
    workers=4,       
    epochs=50        
)


model.wv.save("model_gensim.kv")
print("Modelo guardado con éxito")

print("Palabras similares a 'energizante':")
for word, score in model.wv.most_similar("energizante", topn=5):
    print(f"  - {word}: {score:.4f}")
print("\nPalabras similares a 'terroso':")
for word, score in model.wv.most_similar("terroso", topn=5):
    print(f"  - {word}: {score:.4f}")
