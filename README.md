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
├── data/
│   └── raw/
│       └── cyberbullying_tweets.csv  # Dataset principal
├── src/                              # Módulos Python
│   ├── __init__.py
│   ├── text_preprocessing.py         # Limpieza y preparación de texto
│   ├── vectorization.py              # Vectorización TF-IDF y BoW
│   ├── models.py                     # Modelos ML
│   └── utils.py                      # Funciones de utilidad
├── notebooks/
│   └── cyberbullying_detection.ipynb # Notebook principal interactivo
├── results/                          # Resultados y visualizaciones
│   ├── model_comparison.png
│   ├── train_vs_test_gap.png
│   ├── confusion_matrix_best.png
│   ├── model_metrics.csv
│   ├── classification_report_best.txt
│   ├── examples_correct_incorrect.txt
│   ├── hyperparameter_tuning.csv
│   ├── train_vs_test_comparison.csv
│   ├── user_test_predictions.txt
│   └── run_manifest.json
├── run_project.py                    # Pipeline reproducible principal
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

El proyecto incluye `run_project.py` en la raíz. Este script ejecuta el pipeline completo,
realiza tuning de hiperparámetros con validación cruzada para modelos clásicos estables,
compara train vs test y genera artefactos en `results/`.
La corrida reproducible actual usa vectorización TF-IDF híbrida (palabras + n-gramas de caracteres)
y una rama dirigida de keywords con peso configurable.

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
Se implementan 3 métodos:

**Bag of Words (BoW)**:
- CountVectorizer
- Cuenta frecuencia de palabras
- Útil para baseline

**TF-IDF**:
- TfidfVectorizer (usado para modelos finales)
- Considera frecuencia de término e inversa del documento
- Mejor para clasificación de texto
- N-gramas: unigramas y bigramas
- Max features base de la corrida reproducible: 2,000

**TF-IDF Híbrido (corrida reproducible actual)**:
- Combina TF-IDF de palabras con TF-IDF de caracteres (`char_wb`)
- Captura mejor variaciones ortográficas y patrones subléxicos comunes en acoso
- Se mantiene dentro de modelos estadísticos tradicionales (sin preentrenados)

**TF-IDF Híbrido + Keywords Dirigidas (corrida reproducible actual)**:
- Agrega una rama de patrones léxicos explícitos (ej. `go back your country`, `not want you`)
- Asigna mayor peso relativo a señales dirigidas para mejorar casos ambiguos
- Mantiene el enfoque clásico sin modelos preentrenados

**Script**: `src/vectorization.py`

### 4. División Train-Test
- Split 80-20 (80% training, 20% test)
- Stratified split para mantener distribución de clases
- Asegura balance en ambos conjuntos

### 5. Modelos Implementados y Corrida Reproducible

Modelos implementados en el proyecto:

| Modelo | Algoritmo | Características |
|--------|-----------|-----------------|
| **Naive Bayes** | MultinomialNB | Probabilístico, excelente para texto, muy rápido |
| **Logistic Regression** | LogisticRegression | Lineal, interpretable, buena baseline |
| **SVM** | LinearSVC | Kernel-based, poderoso, bueno en texto |
| **Gradient Boosting** | GradientBoostingClassifier | Ensemble, gran precisión, puede overfitting |
| **Neural Network** | MLPClassifier | Deep learning, 3 capas ocultas (256-128-64) |

**Script**: `src/models.py`

Modelos evaluados en la corrida reproducible (`run_project.py`):

| Modelo | Tipo |
|--------|------|
| **Naive Bayes (tuned)** | Probabilístico + tuning manual con CV |
| **Linear SVM (tuned)** | Lineal + tuning manual con CV |
| **Random Forest** | Ensamble (entrenado en subset estratificado por costo) |

### 6. Evaluación y Métricas

Se usan las siguientes métricas:

- **Accuracy**: Proporción general de predicciones correctas
- **Precision**: De todas las predicciones positivas, cuántas son correctas
- **Recall**: De todos los positivos reales, cuántos detectó
- **F1-Score**: Media armónica de Precision y Recall
- **Weak-Class F1 Avg**: Promedio de F1 para `not_cyberbullying` y `other_cyberbullying`
- **Balance Score**: Criterio final de selección del mejor modelo en corrida reproducible: 0.5 * F1 ponderado + 0.5 * Weak-Class F1 Avg
- **Confusion Matrix**: Matriz de confusión por clase
- **Classification Report**: Reporte detallado por categoría

**Script**: `src/models.py`

## Resultados de la Corrida Reproducible

### Métricas (test)
```
Model                  accuracy  precision  recall    f1      weak_f1_avg  balance_score
Linear SVM (tuned)     0.8182    0.8187     0.8182  0.8182    0.5693       0.6938
Random Forest          0.7981    0.8189     0.7981  0.8024    0.5379       0.6702
Naive Bayes (tuned)    0.7606    0.7563     0.7606  0.7557    0.5083       0.6320
```

### Archivos Generados

1. **model_comparison.png** - Gráfica comparativa de modelos
2. **train_vs_test_gap.png** - Brecha train-test para análisis de overfitting
3. **confusion_matrix_best.png** - Matriz de confusión del mejor modelo
4. **model_metrics.csv** - Métricas en formato tabla
5. **train_vs_test_comparison.csv** - Análisis train vs test
6. **hyperparameter_tuning.csv** - Mejor combinación de hiperparámetros por modelo tuneado
7. **classification_report_best.txt** - Reporte detallado del mejor modelo
8. **examples_correct_incorrect.txt** - Ejemplos bien y mal clasificados
9. **user_test_predictions.txt** - Pruebas de usuario con tweets de entrada
10. **run_manifest.json** - Evidencia de reproducibilidad (seed, dataset, tiempos)

## Función de Predicción

El notebook incluye una función `predict_tweet()` para clasificar tweets nuevos:

```python
# Usar el modelo entrenado para predecir
label, cleaned = predict_tweet("Your tweet here")

print(f"Cleaned: {cleaned}")
print(f"Predicción: {label}")
```

### Salida Esperada
```
Cleaned: stupid nobody like
Predicción: other_cyberbullying
```

## Celdas del Notebook Jupyter

El notebook principal contiene las siguientes secciones:

1. **Import Required Libraries** - Librerías necesarias
2. **Load and Explore the Dataset** - Exploración de datos
3. **Data Cleaning and Preprocessing** - Limpieza de texto NLP
4. **Text Normalization and Tokenization** - Normalización
5. **Text Vectorization** - BoW y TF-IDF
6. **Train-Test Split** - División de datos
7. **Model Training** - Entrenamiento de 3 modelos
8. **Hyperparameter Tuning (CV=2)** - Ajuste manual para Naive Bayes y Linear SVM
9. **Model Comparison** - Comparación de resultados
10. **Confusion Matrix Analysis** - Análisis de errores
11. **User Testing Function** - Predicciones interactivas
12. **Summary and Conclusions** - Resumen final

## Análisis de Resultados

### Interpretación de Métricas

- **Accuracy**: Métrica global, útil para datasets balanceados ✓
- **Precision**: Evita falsos positivos (importante aquí)
- **Recall**: Detecta máximo ciberacoso (importante aquí)
- **F1-Score**: Balance entre ambas (métrica principal usada)

### Overfitting

Se analiza la diferencia train-test por métrica (accuracy, precision, recall, F1, macro F1 y balance score).

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
- Cross-validation explícita manual (cv=2) para modelos lineales/probabilísticos
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
