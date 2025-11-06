
# Cyberbullying PT — TF-IDF + SVM Calibrado

**Meta:** Macro-F1 ≥ 0,70 no teste (por_bully vs por_nonbully).  
**Resultado final:** Macro-F1 **0.7101**, Acc 0.7401 (F1 bully 0.617; nonbully 0.803).

## Dados
Recorte PT do dataset multilíngue (6723 textos; 4604 nonbully / 2119 bully).  
Splits estratificados (seed=42): 70% train, 15% val, 15% test.

## Como reproduzir
```bash
# (opcional) criar venv e instalar deps
pip install -r requirements.txt

# treinar (SVM calibrado)
python -m experiments.01e_train_svc_calibrated_thresh_pt \
  --csv dataset.csv \
  --out runs/pt_svmcal_rf60_C3_ng37 \
  --word_min_df 1 --word_max_df 0.90 \
  --char_min_df 1 --char_ng_min 3 --char_ng_max 7 \
  --C 3.0 --recall_floor 0.60

# gerar CSV de predições do TEST
python scripts/make_test_predictions.py

# (opcional) escanear threshold no VAL
python scripts/scan_threshold_val.py
```
Artefatos principais:

* `runs/pt_svmcal_rf60_C3_ng37/por_test_metrics.json`
* `runs/pt_svmcal_rf60_C3_ng37/por_test_cm.png`
* `runs/pt_svmcal_rf60_C3_ng37/proba_hist_test.png`
* `runs/pt_svmcal_rf60_C3_ng37/por_test_predictions.csv`

## Método

TF-IDF **word(1–2)** + **char_wb(3–7)** → **LinearSVC** calibrado (Platt).
Threshold **t=0.33** escolhido em validação para maximizar Macro-F1 com piso de recall na classe positiva.

## Resultados

| Modelo        | Test Macro-F1 | Acc    | F1(bully) | F1(nonbully) |
| ------------- | ------------- | ------ | --------- | ------------ |
| SVM calibrado | **0.7101**    | 0.7401 | 0.617     | 0.803        |

## Análise de erro

3 FNs e 3 FPs do `por_test_predictions.csv` com comentários breves (ironia/negação; eventos/religião neutros; gíria).

## Ética e limitações

Uso apenas assistivo; possíveis vieses; cuidado com falso positivo; sarcasmo e contexto implícito ainda desafiadores.

## Vídeo

Link: <coloque aqui>

```

---


