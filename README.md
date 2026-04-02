# Universidad del Valle de Guatemala

## Inteligencia Artificial - Proyecto 1

### Integrantes

- Javier España #23361
- Ángel Esquit #23221
- Roberto Barreda #23354

## Tema

Detección automatica de cyberbullying en tweets usando NLP y modelos de Machine Learning.

## Dataset

- Archivo: `data/raw/cyberbullying_tweets.csv`
- Registros: 47,692
- Clases: `age`, `ethnicity`, `gender`, `religion`, `other_cyberbullying`, `not_cyberbullying`

## Requisitos

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python run_project.py
```

## Salidas principales

- `results/figures/`: `model_comparison.png`, `train_vs_test_gap.png`, `confusion_matrix_best.png`, `data_class_distribution.png`
- `results/tables/`: `model_metrics.csv`, `vectorizer_comparison.csv`, `train_vs_test_comparison.csv`, `hyperparameter_tuning.csv`, `common_confusions_best.csv`
- `results/reports/`: `classification_report_best.txt`, `examples_correct_incorrect.txt`
- `results/predictions/`: `user_test_predictions.txt`
- `results/manifests/`: `run_manifest.json`, `data_overview.json`

## Nota

La última corrida selecciona como mejor modelo a **Linear SVM (tuned)** y guarda toda la evidencia en `results/`.

Repositorio: https://github.com/Javier-Espana/PRY1-IA.git