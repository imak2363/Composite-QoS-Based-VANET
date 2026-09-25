# Composite QoS-Based VANET Routing Protocol Selection using Machine Learning

Simulation pipeline, dataset, and machine learning code for selecting the best-performing routing protocol in a Vehicular Ad Hoc Network (VANET) using a composite, multi-metric quality-of-service (QoS) label rather than a single-metric (packet-delivery-ratio-only) baseline.

## Overview

Three VANET routing protocols, AODV, DSDV, and AOMDV, are compared under 3,240 simulated network conditions generated from a real road network extract of Dhaka, Bangladesh. Each condition is simulated once under all three protocols, for 9,720 total simulation runs, varying node count, vehicle speed, flow count, and packet rate.

Two ground-truth labelling schemes are built from the resulting measurements and compared:

- **Single-metric baseline**: the protocol with the highest packet delivery ratio under a given condition.
- **Composite score**: packet delivery ratio, end-to-end delay, jitter, throughput, and normalized routing load, each min-max normalized within its own three-protocol comparison group, then combined using fixed weights.

Five classifiers (logistic regression, decision tree, random forest, XGBoost, and a stacking ensemble) are tuned and compared under three class-imbalance strategies (none, class weighting, SMOTE) for both labelling schemes. The best composite-scheme and PDR-only models are then explained using SHAP and LIME to identify which simulation parameters drive each protocol recommendation.

## Pipeline

1. **Mobility generation**: an OpenStreetMap (OSM) extract of the Dhaka road network is converted to a SUMO network, vehicle mobility traces are simulated in SUMO, and MOVE converts those traces into a format NS-2.35 can read.
2. **Network simulation**: NS-2.35 simulates packet routing under each of the three protocols for every network condition, producing raw QoS trace data.
3. **Label construction** (`build_modeling_table.py`): builds the composite and PDR-only labels from the raw simulation output, with per-condition (per three-protocol group) normalization of each QoS metric.
4. **Classification** (`run_classification_pipeline.py`): trains and tunes all five classifiers under all three imbalance strategies for both labelling schemes, and evaluates them on a held-out test split.
5. **Explainability** (`run_explainability_v2.py`): computes SHAP values and LIME local explanations for the strongest tree-based model on the composite scheme.

See `VANET_DATASET_BUILD_ALL_COMMANDS.txt` for the full step-by-step commands used to reproduce the simulation stage, from OSM map validation through NS-2 batch execution and final dataset verification.

## Repository structure

```
OSM/                                  OpenStreetMap road-network extract used for mobility generation
SUMO/                                 SUMO simulation configuration and mobility traces
MOVE/                                 MOVE tool files converting SUMO traces to NS-2 mobility format
NS2/                                  NS-2.35 simulation scripts and trace output
Dataset/                              Raw and intermediate QoS simulation datasets
Data Results/                         Processed results from the classification and explainability stages
VANET_DATASET_BUILD_ALL_COMMANDS.txt  Step-by-step commands to reproduce the simulation pipeline
build_modeling_table.py               Builds composite and PDR-only labels from raw simulation output
modeling_table_expanded.csv           Modeling table: engineered features plus both label schemes
run_classification_pipeline.py        Trains and evaluates all classifier/imbalance/scheme combinations
run_explainability_v2.py              SHAP and LIME explainability for the best tree-based model
```

## Requirements

- Python 3.x with pandas, numpy, scikit-learn, xgboost, imbalanced-learn, shap, lime, and matplotlib
- SUMO, NS-2.35, and MOVE for reproducing the simulation stage from scratch (not required if working from `modeling_table_expanded.csv` directly)

## Usage

To reproduce the full pipeline from raw simulation output:

```bash
python build_modeling_table.py
python run_classification_pipeline.py
python run_explainability_v2.py
```

To reproduce the network simulation itself, follow the steps in `VANET_DATASET_BUILD_ALL_COMMANDS.txt`.

## Author

Mohammad Kamrul Hasan, Department of Information and Communication Engineering, Noakhali Science and Technology University (NSTU), Bangladesh

## License

Not yet specified.
