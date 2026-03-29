# PROYECTO DE IA 2026 - OPCION B: DETECCION DE CIBERACOSO

## ESTADO ACTUAL

Proyecto actualizado con evidencia reproducible de ejecucion.

- Script principal: `run_project.py`
- Dataset usado: `data/raw/cyberbullying_tweets.csv`
- Semilla de reproducibilidad: 42
- Muestras procesadas: 47,692
- Clases: 6
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

El pipeline incluye preprocesamiento NLP, vectorizacion TF-IDF y comparacion de modelos con evaluacion en test, analisis train-vs-test y tuning de hiperparametros.

---

## ALCANCE IMPLEMENTADO

### 1) Preprocesamiento de texto (NLP)

Implementado en `src/text_preprocessing.py`:
- Eliminacion de URLs, menciones, hashtags, caracteres especiales y emojis
- Normalizacion a minusculas
- Tokenizacion
- Eliminacion de stopwords
- Lematizacion

### 2) Vectorizacion

Implementado en `src/vectorization.py`:
- Bag of Words (BoW)
- TF-IDF (usado en la corrida principal)
- N-gramas (1,2)
- Max features: 5000

### 3) Modelos

- Modelos implementados en el proyecto (`src/models.py`):
  - Naive Bayes
  - Logistic Regression
  - SVM
  - Gradient Boosting
  - MLP

- Modelos evaluados en la corrida reproducible (`run_project.py`):
  - Naive Bayes
  - Logistic Regression (tuned)
  - Gradient Boosting (tuned)

### 4) Evaluacion y analisis

Implementado y guardado en archivos:
- Metricas en test: accuracy, precision, recall, f1
- Comparacion train vs test por metrica
- Matriz de confusion del mejor modelo
- Reporte de clasificacion
- Ejemplos bien/mal clasificados
- Pruebas de usuario

### 5) Tuning de hiperparametros

Se aplico GridSearchCV a:
- Logistic Regression
- Gradient Boosting

---

## RESULTADOS DE LA ULTIMA CORRIDA

Fuente: `results/model_metrics.csv`, `results/hyperparameter_tuning.csv`, `results/run_manifest.json`

### Mejor modelo

- Mejor F1 ponderado: Logistic Regression (tuned)
- F1 test: 0.8134

### Metricas (test)

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression (tuned) | 0.8113 | 0.8166 | 0.8113 | 0.8134 |
| Gradient Boosting (tuned) | 0.8125 | 0.8291 | 0.8125 | 0.8101 |
| Naive Bayes | 0.7568 | 0.7493 | 0.7568 | 0.7478 |

### Mejores hiperparametros encontrados

- Logistic Regression: `{ "C": 1.0, "class_weight": "balanced" }`
- Gradient Boosting: `{ "learning_rate": 0.1, "max_depth": 5, "n_estimators": 150 }`

### Overfitting (train vs test)

La brecha train-test en F1 para los modelos evaluados ronda ~0.054 a ~0.058. 
Se incluyo tabla y grafica para justificar el nivel de generalizacion.

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
- Comparacion de al menos 3 arquitecturas
- Tuning de hiperparametros (2 modelos)
- Tablas de metricas
- Comparacion train vs test
- Matriz de confusion
- Ejemplos bien/mal clasificados
- Funcion de prueba de usuario
- Entrenamiento desde cero (sin modelos preentrenados)

### Pendiente para cierre academico total

- Integrar esta evidencia en el reporte PDF final (sin codigo).
- Incluir justificacion final del mejor modelo balanceando precision y costo computacional en el documento final.

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

---

## NOTAS

- No se usaron modelos preentrenados.
- La ejecucion es reproducible con semilla 42.
- Este archivo refleja el estado real del repositorio al 29 de marzo de 2026.

---

## HANDOFF

### Objetivo inmediato

Completar el cierre academico final con reporte PDF y justificacion tecnica del modelo elegido.

### Siguientes tareas prioritarias

1. Generar el reporte PDF final (sin codigo) usando la evidencia ya creada en `results/`.
2. Incluir una justificacion del mejor modelo balanceando desempeno y costo computacional.
3. Mejorar clases mas debiles (`not_cyberbullying` y `other_cyberbullying`) con una segunda corrida experimental.
4. Dejar una tabla comparativa final entre corrida actual y corrida mejorada.

### Criterio de terminado por tarea

1. Reporte PDF:
  - Incluye EDA, pipeline, particion train/test, modelos, tuning, metricas, matriz de confusion y pruebas de usuario.
  - No contiene codigo fuente.
2. Justificacion tecnica:
  - Explica por que el modelo final es superior.
  - Incluye metricas y costo (tiempo de entrenamiento e inferencia).
3. Mejora de clases debiles:
  - Presenta nuevas metricas por clase.
  - Muestra si subio F1 de `not_cyberbullying` y `other_cyberbullying`.
4. Cierre de entrega:
  - ZIP/RAR con codigo y recursos.
  - PDF por separado, listo para subir.

### Comandos de trabajo rapido

1. Instalar dependencias:
  - `pip install -r requirements.txt`
2. Ejecutar pipeline base:
  - `python run_project.py`
3. Revisar artefactos:
  - Carpeta `results/`

### Tiempos esperados

- Ejecucion de `run_project.py`: ~8 a 12 minutos en equipo promedio.
- Redaccion y armado de PDF final: 1 a 2 horas.
- Iteracion de mejora de clases debiles: 1 a 3 horas (segun cantidad de pruebas).

### Riesgos conocidos

1. Rendimiento desigual por clase: las clases `not_cyberbullying` y `other_cyberbullying` son las mas dificiles.
2. Sobreajuste moderado: gap train-test alrededor de ~0.054 a ~0.058.
3. Posibles errores NLTK en entorno nuevo: si falta recurso, descargar `punkt_tab`.

### Archivos clave para continuar

1. Script de ejecucion: `run_project.py`
2. Resumen de estado: `PROJECT_SUMMARY.md`
3. Metricas globales: `results/model_metrics.csv`
4. Metricas train-test: `results/train_vs_test_comparison.csv`
5. Tuning: `results/hyperparameter_tuning.csv`
6. Reporte por clase: `results/classification_report_best.txt`
