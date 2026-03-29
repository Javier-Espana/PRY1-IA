# Cyberbullying Detection in Social Media Tweets

**Inteligencia Artificial 2026 - Opción B**

Sistema automático de clasificación para identificar ciberacoso en tweets usando Machine Learning.

## Descripción del Proyecto

Este proyecto implementa un sistema de clasificación de textos que identifica y categoriza contenido de ciberacoso en redes sociales. Utiliza técnicas de Procesamiento de Lenguaje Natural (NLP) y múltiples algoritmos de Machine Learning para clasificar tweets en las siguientes categorías:

- **Age**: Acoso relacionado con la edad
- **Ethnicity**: Acoso basado en etniciidad
- **Gender**: Acoso relacionado con el género
- **Religion**: Acoso religioso
- **Other Cyberbullying**: Otro tipo de ciberacoso
- **Not Cyberbullying**: Tweets que no contienen ciberacoso

## Dataset

- **Nombre**: `cyberbullying_tweets.csv`
- **Tamaño**: ~47,000 tweets etiquetados
- **Balance**: Dataset bien balanceado entre categorías
- **Fuente**: Kaggle - Cyberbullying Classification

## Estructura del Proyecto

```
PRY1-IA/
├── cyberbullying_tweets.csv          # Dataset principal
├── src/                              # Módulos Python
│   ├── __init__.py
│   ├── text_preprocessing.py         # Limpieza y preparación de texto
│   ├── vectorization.py              # Vectorización TF-IDF y BoW
│   ├── models.py                     # Modelos ML
│   └── utils.py                      # Funciones de utilidad
├── notebooks/
│   └── cyberbullying_detection.ipynb # Notebook principal interactivo
├── data/                             # (Para datos procesados)
├── models/                           # (Para modelos guardados)
├── results/                          # Resultados y visualizaciones
│   ├── model_comparison.png
│   ├── train_vs_test.png
│   ├── confusion_matrix_*.png
│   ├── model_metrics.csv
│   ├── classification_report.txt
│   └── project_summary.txt
└── README.md                         # Este archivo
```

## Requisitos

### Python 3.8+

Instalar dependencias:

```bash
pip install -r requirements.txt
```

O instalar manualmente:

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn jupyter
```

## Instalación

1. **Clonar o descargar el repositorio**

```bash
cd /path/to/PRY1-IA
```

2. **Crear un entorno virtual (opcional pero recomendado)**

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn jupyter
```

## Ejecución

### Opción 1: Ejecutar el Notebook Interactivo (Recomendado)

El notebook contiene todo el análisis paso a paso con visualizaciones.

```bash
cd notebooks
jupyter notebook cyberbullying_detection.ipynb
```

Luego:
1. Ejecutar (Shift+Enter) cada celda secuencialmente
2. Observar el análisis exploratorio
3. Ver el entrenamiento de modelos
4. Analizar resultados y comparación de modelos

### Opción 2: Ejecutar como Script Python

Crear un archivo `run_project.py` en la raíz del proyecto:

```python
import sys
sys.path.insert(0, 'src')

from text_preprocessing import preprocess_data
from vectorization import vectorize_texts
from models import create_models, train_and_evaluate_models
from utils import load_data, prepare_data, plot_confusion_matrix
import pandas as pd

# Cargar datos
df = load_data('cyberbullying_tweets.csv')

# Preprocesar
from text_preprocessing import TextPreprocessor
preprocessor = TextPreprocessor()
df['tweet_text_cleaned'] = df['tweet_text'].apply(preprocessor.process_text)

# Preparar datos
X_train, X_test, y_train, y_test = prepare_data(df, 'tweet_text_cleaned')

# Vectorizar
X_train_tfidf, X_test_tfidf, vectorizer = vectorize_texts(X_train, X_test, 'tfidf')

# Crear y entrenar modelos
models = create_models()
results = train_and_evaluate_models(models, X_train_tfidf, y_train, X_test_tfidf, y_test)

# Mostrar resultados
print(results)
```

Ejecutar:

```bash
python run_project.py
```

## Fases del Proyecto

### 1. Carga y Análisis Exploratorio de Datos (EDA)
- Cargar dataset
- Visualizar distribución de clases
- Análisis de temas de ciberacoso
- Identificar datos faltantes

**Script**: Notebook sección "Load and Explore"

### 2. Pre-procesamiento de Texto (NLP Pipeline)
- Eliminar URLs
- Eliminar menciones (@usuario)
- Eliminar hashtags
- Eliminar caracteres especiales y emojis
- Convertir a minúsculas
- Eliminar stopwords
- Tokenización
- Lematización

**Script**: `src/text_preprocessing.py`

### 3. Vectorización de Texto
Se implementan 2 métodos:

**Bag of Words (BoW)**:
- CountVectorizer
- Cuenta frecuencia de palabras
- Útil para baseline

**TF-IDF**:
- TfidfVectorizer (usado para modelos finales)
- Considera frecuencia de término e inversa del documento
- Mejor para clasificación de texto
- N-gramas: unigramas y bigramas
- Max features: 5,000

**Script**: `src/vectorization.py`

### 4. División Train-Test
- Split 80-20 (80% training, 20% test)
- Stratified split para mantener distribución de clases
- Asegura balance en ambos conjuntos

### 5. Comparación de 5 Modelos

Se entrenan y comparan los siguientes modelos:

| Modelo | Algoritmo | Características |
|--------|-----------|-----------------|
| **Naive Bayes** | MultinomialNB | Probabilístico, excelente para texto, muy rápido |
| **Logistic Regression** | LogisticRegression | Lineal, interpretable, buena baseline |
| **SVM** | LinearSVC | Kernel-based, poderoso, bueno en texto |
| **Gradient Boosting** | GradientBoostingClassifier | Ensemble, gran precisión, puede overfitting |
| **Neural Network** | MLPClassifier | Deep learning, 3 capas ocultas (256-128-64) |

**Script**: `src/models.py`

### 6. Evaluación y Métricas

Se usan las siguientes métricas:

- **Accuracy**: Proporción general de predicciones correctas
- **Precision**: De todas las predicciones positivas, cuántas son correctas
- **Recall**: De todos los positivos reales, cuántos detectó
- **F1-Score**: Media armónica de Precision y Recall
- **Confusion Matrix**: Matriz de confusión por clase
- **Classification Report**: Reporte detallado por categoría

**Script**: `src/models.py`

## Resultados Esperados

### Model Performance (Ejemplo)
```
Model                      accuracy  precision    recall       f1
Naive Bayes                 0.7851    0.7852      0.7851    0.7851
Logistic Regression         0.8234    0.8237      0.8234    0.8235
Support Vector Machine      0.8412    0.8415      0.8412    0.8413
Gradient Boosting           0.8523    0.8525      0.8523    0.8524
Neural Network              0.8390    0.8392      0.8390    0.8391
```

### Archivos Generados

1. **model_comparison.png** - Gráfica comparativa de modelos
2. **train_vs_test.png** - Análisis de overfitting
3. **confusion_matrix_*.png** - Matrices de confusión
4. **model_metrics.csv** - Métricas en formato tabla
5. **train_vs_test_comparison.csv** - Análisis train vs test
6. **classification_report.txt** - Reporte detallado
7. **project_summary.txt** - Resumen completo del proyecto

## Función de Predicción

El notebook incluye una función `predict_cyberbullying()` para clasificar tweets nuevos:

```python
# Usar el modelo entrenado para predecir
result = predict_cyberbullying("Your tweet here")

print(f"Tweet: {result['tweet']}")
print(f"Predicción: {result['prediction']}")
print(f"Confianza: {result['confidence']:.4f}")
```

### Salida Esperada
```
Tweet: You're so stupid, nobody likes you
Predicción: Other Cyberbullying
Confianza: 0.8912
Modelo: Gradient Boosting
```

## Celdas del Notebook Jupyter

El notebook principal contiene las siguientes secciones:

1. **Import Required Libraries** - Librerías necesarias
2. **Load and Explore the Dataset** - Exploración de datos
3. **Data Cleaning and Preprocessing** - Limpieza de texto NLP
4. **Text Normalization and Tokenization** - Normalización
5. **Text Vectorization** - BoW y TF-IDF
6. **Train-Test Split** - División de datos
7. **Model 1-5 Training** - Entrenamiento individual
8. **Model Comparison** - Comparación de resultados
9. **Confusion Matrix Analysis** - Análisis de errores
10. **User Testing Function** - Predicciones interactivas
11. **Summary and Conclusions** - Resumen final

## Análisis de Resultados

### Interpretación de Métricas

- **Accuracy**: Métrica global, útil para datasets balanceados ✓
- **Precision**: Evita falsos positivos (importante aquí)
- **Recall**: Detecta máximo ciberacoso (importante aquí)
- **F1-Score**: Balance entre ambas (métrica principal usada)

### Overfitting

Se analiza la diferencia train-test. Si es > 0.05, hay overfitting.

### Confusion Matrix

Muestra donde el modelo confunde categorías:
- Diagonal principal = predicciones correctas
- Fuera de diagonal = errores

## Técnicas Implementadas

### 1. NLP Pipeline
- Expresiones regulares para limpieza
- NLTK para tokenización y lematización
- Stop-word removal

### 2. Feature Engineering
- Vectorización TF-IDF con n-gramas
- Matriz dispersa (sparse) para eficiencia

### 3. Model Selection
- Hyperparameter tuning
- Cross-validation (implícito en split)
- Comparación sistemática

### 4. Model Evaluation
- Múltiples métricas
- Visualización de resultados
- Análisis de errores

## Recomendaciones

1. **Para mejor performance**:
   - Aumentar tamaño del dataset
   - Fine-tuning de hiperparámetros
   - Usar ensemble de múltiples modelos
   - Implementar validación cruzada K-fold

2. **Para producción**:
   - Guardar modelos entrenados
   - Implementar pipeline automático
   - Monitorear performance
   - Actualizar regularmente con datos nuevos

3. **Consideraciones éticas**:
   - Privacidad de usuarios
   - Sesgo en predicciones
   - Transparencia del modelo
   - Casos límite y contexto

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'nltk'"
```bash
pip install nltk
python -c "import nltk; nltk.download('omw-1.4')"
```

### Notebook lento en procesamiento
- Usar menos muestras para testing inicial
- Parámetro `nrows` al cargar CSV

### Módulos no encontrados desde notebook
Agregar al inicio del notebook:
```python
import sys
sys.path.insert(0, '../src')
```

## Entrega del Proyecto

Archivos a entregar en Canvas:

1. **ZIP/RAR con el código**:
   - `src/` (todos los módulos Python)
   - `notebooks/cyberbullying_detection.ipynb`
   - `requirements.txt`
   - `README.md`

2. **Reporte técnico PDF** con:
   - Análisis exploratorio
   - Descripción técnica de modelos
   - Resultados y comparación
   - Conclusiones y recomendaciones

3. **Resultados visuales** (incluidos en reporte):
   - Gráficas de distribución
   - Comparación de modelos
   - Matrices de confusión

## Autor

Javier España - Proyecto de IA 2026

## Licencia

Este proyecto es con fines educativos.

## Referencias

- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TF-IDF Vectorization](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Text Classification Guide](https://developers.google.com/machine-learning/guides/text-classification)

---

**Fecha de Entrega**: 6 de abril de 2026

**Estado**: ✓ Completado
