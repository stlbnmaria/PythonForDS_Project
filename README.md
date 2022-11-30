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
 - Covid cases (national daily covid cases in France (see detailed description in [dedicated notebook](data_analyses/covid_data.ipynb))

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

Below you can find the output for the tuned Catboost model. It seems to be the best performing algorithm for this project. The model had the lowest RMSE in GridSearchCV compared to all trained models. The mean validation score in GridSearchCV was 0.73. For the ramp-test, the bagged validation score was 0.735 and the bagged test score was 0.539. 

<details>

<summary>Output of the best model</summary>

```
Testing Bike count prediction
Reading train and test files from ./data/ ...
Reading cv ...
Training submissions/221129_lowest_test ...
CV fold 0
        score   rmse      time
        train  0.289  8.144703
        valid  0.885  1.524310
        test   0.628  0.274136
CV fold 1
        score   rmse      time
        train  0.327  9.824058
        valid  0.738  1.383957
        test   0.583  0.259836
CV fold 2
        score   rmse       time
        train  0.340  12.709411
        valid  0.673   1.366187
        test   0.541   0.261274
CV fold 3
        score   rmse       time
        train  0.363  16.143719
        valid  0.558   1.357172
        test   0.663   0.261925
CV fold 4
        score   rmse       time
        train  0.371  25.455985
        valid  0.694   1.707904
        test   0.611   0.316238
CV fold 5
        score   rmse       time
        train  0.381  23.015316
        valid  0.710   1.304126
        test   0.625   0.258692
CV fold 6
        score   rmse       time
        train  0.387  28.040693
        valid  0.852   1.358390
        test   0.603   0.257964
CV fold 7
        score   rmse       time
        train  0.386  31.464994
        valid  0.723   1.352725
        test   0.746   0.267434
----------------------------
Mean CV scores
----------------------------
        score            rmse         time
        train  0.355 ± 0.0323  19.3 ± 8.24
        valid  0.729 ± 0.0958   1.4 ± 0.12
        test   0.625 ± 0.0565   0.3 ± 0.02
----------------------------
Bagged scores
----------------------------
        score   rmse
        valid  0.735
        test   0.539
```

</details>
