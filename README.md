# Paris Bike Counters Project

Date: 28.11.2022

Contributors: João Melo and Maria Stoelben

## Context and description of the project

We are two students of the M.Sc. Data Science for Business at École Polytechnique and HEC Paris. This project was created for a course in our first year. The aim is to predict the bicycle traffic for 30 different bike counters in Paris. The target variable is the number of bikes passing by for every hour. Furthermore, the target variable was transformed by $y'=log(1+y)$ to reduce skewness. Our data analyses can be found in this [folder](data_analyses). The different models, tuning scripts and aggregation can be found [here](modeling). Final submissions for the ramp platform and their encoded names can be found [here](submission).

## Data 
 -  Bike counters data    
    - [train.parquet](https://github.com/ramp-kits/bike_counters/releases/download/v0.1.0/train.parquet)
    - [test.parquet](https://github.com/ramp-kits/bike_counters/releases/download/v0.1.0/test.parquet)
 - Paris weather data (see detailed description in [dedicated notebook](data_analyses/weather_data_analyses.ipynb))
 - Public holidays calendar ([jours-feries-france 0.7.0](https://pypi.org/project/jours-feries-france/))
 - School holidays calendar ([vacances-scolaires-france 0.9.0](https://pypi.org/project/vacances-scolaires-france/))
 - Covid lockdowns (national lockdowns indicated [here](https://en.wikipedia.org/wiki/COVID-19_pandemic_in_France))

## Algorithms 
We worked with the following algorithms. Tuning was done by using an 8-fold cross validation time series split.

- Linear Models
    - Ridge
    - Lasso
- Geometric Models
    - KNN
- Kernel Models 
    - Linear SVR
- Tree Models
    - Random forest
    - Boosting
        - XGB
        - Catboost
        - LightGBM
    - Extremely randomized Trees
    - LCE Ensemble
- Stacked Generalization
    - Voting Regressor

## Results

Due to readability, we display only the best performing models compared to a baseline in the following graph. All of the final models are tuned and the Voting Regressor is the weighted average of the the Catboost, LightGBM and XGB model.

![Alt text](modeling/scores_comparison.png?raw=true "Title")

Below you can find the output for the tuned Catboost model. It seems to be the best performing algorithm for this project. The model had the lowest RMSE in GridSearchCV compared to all trained models. The mean validation score in GridSearchCV was 0.71. For the ramp-test, the bagged validation score was 0.73 and the bagged test score was 0.54. 

<details>

<summary>Output of the best model</summary>

```
Testing Bike count prediction
Reading train and test files from ./data/ ...
Reading cv ...
Training submissions/221127_cat_v2data_final ...
CV fold 0
        score   rmse       time
        train  0.287  13.768831
        valid  0.866   1.562365
        test   0.618   0.275989
CV fold 1
        score   rmse       time
        train  0.320  17.759133
        valid  0.721   1.594459
        test   0.588   0.271921
CV fold 2
        score   rmse       time
        train  0.332  19.889750
        valid  0.639   1.440514
        test   0.548   0.270202
CV fold 3
        score   rmse       time
        train  0.355  23.975242
        valid  0.558   1.455431
        test   0.638   0.267357
CV fold 4
        score   rmse       time
        train  0.364  30.052660
        valid  0.687   1.459324
        test   0.599   0.279093
CV fold 5
        score   rmse       time
        train  0.376  33.125565
        valid  0.707   1.469339
        test   0.628   0.277371
CV fold 6
        score   rmse       time
        train  0.381  43.913053
        valid  0.862   1.441575
        test   0.604   0.270822
CV fold 7
        score   rmse       time
        train  0.379  35.565420
        valid  0.704   1.414413
        test   0.694   0.271785
----------------------------
Mean CV scores
----------------------------
        score            rmse         time
        train  0.349 ± 0.0314  27.3 ± 9.53
        valid   0.718 ± 0.097   1.5 ± 0.06
        test   0.615 ± 0.0396    0.3 ± 0.0
----------------------------
Bagged scores
----------------------------
        score   rmse
        valid  0.725
        test   0.543
```

</details>
