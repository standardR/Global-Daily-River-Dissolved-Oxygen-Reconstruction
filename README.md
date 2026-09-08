# Unveiling the Patterns and Drivers of Global Riverine Dissolved Oxygen via a Transformer-based Deep Learning
![Model Architecture](image/img.png)

This project presents a transformer-based deep learning framework to reconstruct daily dissolved oxygen (DO) concentrations in rivers at a global scale by integrating hydrometeorological datasets and watershed attributes. The study addresses the challenges of spatiotemporal gaps in water quality records and provides new insights into the key drivers of DO dynamics worldwide.

## Highlights

- Transformer model effectively captures daily river DO dynamics worldwide.
- 30-day cumulative air temperature is the main driver of global river DO dynamics.
- Atmospheric warming and river deoxygenation are strongly linked globally.
- Agricultural basins show fastest warming and most severe deoxygenation.

## Main Features

- Handling sparse water quality data and large data gaps.
- All input variables are derived from datasets with global coverage.
- Model interpretability for input importance.

## File Descriptions

- **DO_benchmark.py**  
  Benchmark model script for comparing the performance of different models on the dataset.

- **DO_benchmark_test.py**  
  Testing script for the benchmark model, used to evaluate performance on the test set.

- **DOformer.py**  
  Main script defining and training the DOformer model.

- **DOformerTest.py**  
  Testing script for the DOformer model, evaluating its performance on the test set.

- **DOformerVal.py**  
  Validation script for the DOformer model, used for hyperparameter tuning and validation.

- **DOformer_optuna.py**  
  Script for hyperparameter optimization of the DOformer model using Optuna.

- **Google Earth Engine Data Extraction Code.txt**  
  Code for extracting data from Google Earth Engine (GEE).

- **feature_importance_results.csv & global_delta.csv**  
  Results of the feature importance analysis.

- **model_optuna.py**  
  Script for hyperparameter optimization of a general model using Optuna.

- **requirements.txt**  
  Python environment dependency list for installing the required packages.

- **test_predictions.csv**  
  Model predictions on the test set.


## Contact

Kun Shan
Chongqing Institute of Green and Intelligent Technology,  
Chinese Academy of Sciences, Chongqing, 400714, China  
E-mail: shankun@cigit.ac.cn

---
