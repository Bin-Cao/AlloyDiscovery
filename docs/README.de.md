<div align="center">

# [ATON for Alloy Discovery](https://bin-cao.github.io/AlloyDiscovery/)

**Trajektorienbasierte Vorhersage von Legierungseigenschaften mit selbstkonsistenter Inferenz und Residualkorrektur.**

[English](../README.md) | [中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Deutsch](README.de.md)

</div>

Alloy Trajectory Optimization Network/Model (ATON/ATOM) sagt mechanische Eigenschaften von Legierungen aus Zusammensetzungs- und Prozessdeskriptoren vorher:

- `Strength_MPa`
- `Plasticity_%`

Das Modell kombiniert ein neuronales ATOM-Backbone, selbstkonsistente Trajektorieninferenz und einen internen Residualkorrektur-Zweig. Benutzerorientierte Vorhersagedateien enthalten nur die finalen ATOM-Vorhersagespalten.

## Ergebnisvorschau

<p align="center">
  <img src="figs/tensile.png" alt="Vergleich der Festigkeitsvorhersage" width="48%">
  <img src="figs/elongation.png" alt="Vergleich der Plastizitätsvorhersage" width="48%">
</p>

Das Vergleichsnotebook bewertet ATOM gegen RF, GBDT, SVR, Ridge, Lasso, ElasticNet und KNN mit demselben Fünf-Fold-Out-of-Fold-Protokoll.

## Schnellstart

```bash
python trainer.py
python inference.py input.xlsx predictions.xlsx --checkpoint checkpoints/model_best.pth
```

Erzeugte Trainingsartefakte:

```text
checkpoints/model_best.pth
checkpoints/cv_oof_predictions.xlsx
checkpoints/cv_summary.json
```

Finale Inferenzspalten:

```text
ATOM_Predicted_Strength_MPa
ATOM_Predicted_Plasticity_%
```

## Repository-Struktur

```text
data.xlsx                 Trainingsdaten
trainer.py                Einstieg fuer Fuenf-Fold-Cross-Validation-Training
inference.py              Einstieg fuer label-freie Inferenz
build_search_space.py     Virtuellen Suchraum erzeugen
src/data.py               Spaltendefinitionen und Dataset-Erstellung
src/model.py              Neuronales ATOM-Modell
src/ensemble.py           Interner Residualkorrektur-Zweig
src/train.py              Training und selbstkonsistente Vorhersagehilfen
docs/algorithm.html       Detailliertes englisch/chinesisches Algorithmen-Dokument
docs/README.md            Mehrsprachiger README-Umschalter
docs/figs/plot.ipynb      Fuenf-Fold-Vergleichsplots
```

## Datenformat

Trainingsdaten muessen diese Spalten enthalten:

```text
Al, Co, Cr, Fe, Ni, Ti, Mo, Nb
Eutectic, Preparation, Processing, Tensile/compress
Cold CR_%, Annealing_Temp_K, Annealing_time_h, Aging_Temp_K, Aging_Time_h
Rate_S-1, Tes_Temp_K
Strength_MPa, Plasticity_%
```

Inferenzdaten benoetigen nur Feature-Spalten. Falls Label-Spalten vorhanden sind, entfernt `inference.py` sie automatisch.

## Training

Aus dem Projektwurzelverzeichnis ausfuehren:

```bash
python trainer.py
```

Das Cross-Validation-Ergebnis wird berechnet, indem alle fuenf held-out Fold-Vorhersagen konkateniert und einmal auf dem vollstaendigen Out-of-Fold-Vorhersagevektor bewertet werden.

Nuetzliche Optionen:

```bash
python trainer.py \
  --data data.xlsx \
  --output-dir checkpoints \
  --epochs 300 \
  --batch-size 32 \
  --lr 1e-3 \
  --patience 50 \
  --sc-max-iters 30 \
  --sc-tol 1e-3
```

## Inferenz

```bash
python inference.py input.xlsx predictions.xlsx --checkpoint checkpoints/model_best.pth
```

## Suchraum

```bash
python build_search_space.py --data data.xlsx --output search_space.csv
python inference.py search_space.csv search_predictions.xlsx --checkpoint checkpoints/model_best.pth
```

## Modellvergleichsplots

Zuerst das Modell trainieren:

```bash
python trainer.py
```

Dann oeffnen und ausfuehren:

```text
docs/figs/plot.ipynb
```

## Eigene Daten verwenden

1. Eine Excel- oder CSV-Datei mit den erforderlichen Feature-Spalten vorbereiten.
2. `Strength_MPa` und `Plasticity_%` nur in Trainingsdaten behalten.
3. Kategoriale Spalten als nichtnegative Ganzzahlen kodieren.
4. Wenn Element- oder Prozessspalten abweichen, die Spaltenlisten in `src/data.py` bearbeiten.
5. Trainieren mit:

```bash
python trainer.py --data your_data.xlsx
```

## Dokumentation

Vollstaendige Algorithmen-Anleitung:

- [Online-Dokumentation](https://bin-cao.github.io/AlloyDiscovery/)
- [`docs/algorithm.html`](algorithm.html)
