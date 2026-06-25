<div align="center">

# [ATON for Alloy Discovery](https://bin-cao.github.io/AlloyDiscovery/)

**Predicción de propiedades de aleaciones basada en trayectorias con inferencia autoconsistente y corrección residual.**

[English](../README.md) | [中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Español](README.es.md) | [Deutsch](README.de.md)

</div>

Alloy Trajectory Optimization Network/Model (ATON/ATOM) predice propiedades mecánicas de aleaciones a partir de descriptores de composición y procesamiento:

- `Strength_MPa`
- `Plasticity_%`

El modelo combina una red neuronal ATOM, inferencia de trayectoria autoconsistente y una rama interna de corrección residual. Los archivos de predicción para usuarios solo exponen las columnas finales de predicción ATOM.

## Vista previa de resultados

<p align="center">
  <img src="figs/tensile.png" alt="Comparación de predicción de resistencia" width="48%">
  <img src="figs/elongation.png" alt="Comparación de predicción de plasticidad" width="48%">
</p>

El notebook de comparación evalúa ATOM frente a RF, GBDT, SVR, Ridge, Lasso, ElasticNet y KNN bajo el mismo protocolo out-of-fold de cinco particiones.

## Inicio rápido

```bash
python trainer.py
python inference.py input.xlsx predictions.xlsx --checkpoint checkpoints/model_best.pth
```

Artefactos generados por el entrenamiento:

```text
checkpoints/model_best.pth
checkpoints/cv_oof_predictions.xlsx
checkpoints/cv_summary.json
```

Columnas finales de inferencia:

```text
ATOM_Predicted_Strength_MPa
ATOM_Predicted_Plasticity_%
```

## Estructura del repositorio

```text
data.xlsx                 Datos de entrenamiento
trainer.py                Entrada de entrenamiento con validación cruzada de cinco particiones
inference.py              Entrada de inferencia sin etiquetas
build_search_space.py     Generación de espacio de búsqueda virtual
src/data.py               Definiciones de columnas y construcción del dataset
src/model.py              Modelo neuronal ATOM
src/ensemble.py           Rama interna de corrección residual
src/train.py              Utilidades de entrenamiento y predicción autoconsistente
docs/algorithm.html       Documento detallado del algoritmo en inglés/chino
docs/README.md            Selector multilingüe del README
docs/figs/plot.ipynb      Gráficas de comparación de cinco particiones
```

## Formato de datos

Los datos de entrenamiento deben contener estas columnas:

```text
Al, Co, Cr, Fe, Ni, Ti, Mo, Nb
Eutectic, Preparation, Processing, Tensile/compress
Cold CR_%, Annealing_Temp_K, Annealing_time_h, Aging_Temp_K, Aging_Time_h
Rate_S-1, Tes_Temp_K
Strength_MPa, Plasticity_%
```

Los datos de inferencia solo necesitan columnas de características. Si hay columnas de etiquetas, `inference.py` las elimina automáticamente.

## Entrenamiento

Ejecuta desde la raíz del proyecto:

```bash
python trainer.py
```

El resultado de validación cruzada se calcula concatenando las predicciones de las cinco particiones retenidas y evaluando una vez sobre el vector out-of-fold completo.

Opciones útiles:

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

## Inferencia

```bash
python inference.py input.xlsx predictions.xlsx --checkpoint checkpoints/model_best.pth
```

## Espacio de búsqueda

```bash
python build_search_space.py --data data.xlsx --output search_space.csv
python inference.py search_space.csv search_predictions.xlsx --checkpoint checkpoints/model_best.pth
```

## Gráficas de comparación

Primero entrena el modelo:

```bash
python trainer.py
```

Luego abre y ejecuta:

```text
docs/figs/plot.ipynb
```

## Usar tus propios datos

1. Prepara un archivo Excel o CSV con las columnas de características requeridas.
2. Conserva `Strength_MPa` y `Plasticity_%` solo en los datos de entrenamiento.
3. Codifica las columnas categóricas como enteros no negativos.
4. Si tus columnas de elementos o procesos son distintas, edita las listas de columnas en `src/data.py`.
5. Entrena con:

```bash
python trainer.py --data your_data.xlsx
```

## Documentación

Guía completa del algoritmo:

- [Documentación en línea](https://bin-cao.github.io/AlloyDiscovery/)
- [`docs/algorithm.html`](algorithm.html)
