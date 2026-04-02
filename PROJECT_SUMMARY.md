# PROYECTO DE IA 2026 - OPCION B: DETECCION DE CIBERACOSO

## ESTADO ACTUAL

Proyecto actualizado con evidencia reproducible de ejecucion.

- Script principal: `run_project.py`
- Dataset usado: `data/raw/cyberbullying_tweets.csv`
- Semilla de reproducibilidad: 42
- Muestras procesadas: 47,692
- Clases: 6
- Vectorizacion de corrida: `tfidf_hybrid_keywords`
- CV en corrida actual: 2 folds (tuning manual en NB y SVM)
- Ensamble: Random Forest sobre subset estratificado (12,000 muestras de train)
- Ultima corrida: completada sin errores

---

## RESUMEN EJECUTIVO

Se implemento un sistema de clasificacion supervisada para detectar tipos de ciberacoso en tweets.

Categorias objetivo:
- age
- ethnicity
- gender
- religion
- other_cyberbullying
- not_cyberbullying

El pipeline incluye:
- Preprocesamiento NLP clasico
- Vectorizacion TF-IDF hibrida (palabras + caracteres)
- Rama dirigida de keywords de rechazo/xenofobia
- Comparacion de modelos clasicos
- Analisis train-vs-test
- Artefactos de evaluacion completos

---

## ALCANCE IMPLEMENTADO

### 1) Preprocesamiento de texto (NLP)

Implementado en `src/text_preprocessing.py`:
- Eliminacion de URLs, menciones, hashtags, caracteres especiales y emojis
- Normalizacion a minusculas
- Tokenizacion
- Eliminacion de stopwords (preservando negaciones y pronombres clave)
- Lematizacion

### 2) Vectorizacion

Implementado en `src/vectorization.py`:
- Bag of Words (BoW)
- TF-IDF (palabras)
- TF-IDF hibrido (palabras + n-gramas de caracteres)
- TF-IDF hibrido + keywords dirigidas con peso configurable
- Max features base: 2000
- Peso de keywords en corrida actual: 3.0

### 3) Modelos evaluados en corrida reproducible

- Naive Bayes (tuned)
- Linear SVM (tuned)
- Random Forest (ensamble, parametros fijos por estabilidad)

### 4) Evaluacion y analisis

Generado en `results/`:
- Metricas en test: accuracy, precision, recall, f1
- Metricas extendidas: macro_f1, weak_f1_avg, balance_score
- Comparacion train vs test por metrica
- Matriz de confusion del mejor modelo
- Classification report por clase
- Ejemplos bien/mal clasificados
- Pruebas de usuario

### 5) Tuning de hiperparametros

- Naive Bayes: tuning manual con CV=2
- Linear SVM: tuning manual con CV=2
- Random Forest: parametros fijos, entrenado en subset estratificado para costo computacional

---

## RESULTADOS DE LA ULTIMA CORRIDA

Fuente: `results/model_metrics.csv`, `results/hyperparameter_tuning.csv`, `results/run_manifest.json`

### Mejor modelo

- Mejor modelo por criterio balanceado: Linear SVM (tuned)
- Criterio: `balance_score = 0.5 * F1 ponderado + 0.5 * promedio F1(not_cyberbullying, other_cyberbullying)`
- F1 test: 0.8182
- Balance score test: 0.6938

### Metricas (test)

| Modelo | Accuracy | Precision | Recall | F1 | Weak-F1 Avg | Balance Score |
|---|---:|---:|---:|---:|---:|---:|
| Linear SVM (tuned) | 0.8182 | 0.8187 | 0.8182 | 0.8182 | 0.5693 | 0.6938 |
| Random Forest | 0.7981 | 0.8189 | 0.7981 | 0.8024 | 0.5379 | 0.6702 |
| Naive Bayes (tuned) | 0.7606 | 0.7563 | 0.7606 | 0.7557 | 0.5083 | 0.6320 |

### Hiperparametros seleccionados

- Naive Bayes: `{ "alpha": 0.005 }`
- Linear SVM: `{ "C": 0.5, "class_weight": "balanced" }`
- Random Forest (fixed): `{ "n_estimators": 120, "max_depth": 20, "min_samples_split": 2 }`

### Overfitting (train vs test)

- Gap F1 aprox.: 0.023 a 0.054
- Gap balance_score aprox.: 0.041 a 0.089

---

## ARTEFACTOS GENERADOS

Carpeta: `results/`

- `model_metrics.csv`
- `train_vs_test_comparison.csv`
- `hyperparameter_tuning.csv`
- `model_comparison.png`
- `train_vs_test_gap.png`
- `confusion_matrix_best.png`
- `classification_report_best.txt`
- `examples_correct_incorrect.txt`
- `user_test_predictions.txt`
- `run_manifest.json`

---

## CUMPLIMIENTO DE INSTRUCCIONES (OPCION B)

### Cumplido

- Preprocesamiento NLP
- Vectorizacion BoW y TF-IDF
- Comparacion de al menos 3 arquitecturas clasicas
- Tuning de hiperparametros con validacion cruzada (NB y SVM)
- Tablas de metricas
- Comparacion train vs test
- Matriz de confusion
- Ejemplos bien/mal clasificados
- Funcion de prueba de usuario
- Entrenamiento desde cero (sin modelos preentrenados)

### Nota de implementacion

La funcion de prueba de usuario incluye una capa transparente de reglas para patrones explicitos de rechazo/xenofobia. Esta capa solo afecta predicciones de usuario y no altera las metricas del benchmark de modelos.

---

## COMO REPRODUCIR

1. Instalar dependencias:

```bash
pip install -r requirements.txt
```

2. Ejecutar pipeline completo:

```bash
python run_project.py
```

3. Revisar resultados en `results/`.
