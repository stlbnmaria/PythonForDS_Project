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

Due to readability, we display only the best performing models compared to a baseline in the following graph. All of the final models are tuned and the Voting Regressor is the weighted average of the Catboost (with the lowest test RMSE), LightGBM and XGB model. The best model is not inluded in the Voting Regressor, since the Stacked Generalization did not seem to improve performance while also being more computationally expensive.

![Alt text](modeling/scores_comparison.png?raw=true "Title")

Below you can find the output for the tuned Catboost model with parts of all external data sources. It seems to be the best performing algorithm for this project. The mean validation score in GridSearchCV was 0.73. For the ramp-test, the bagged validation score was 0.73 and the bagged test score was 0.55. 

<details>

<summary>Output of the best model</summary>

```
Testing Bike count prediction
Reading train and test files from ./data/ ...
Reading cv ...
Training submissions/221129_test_new_data_5 ...
CV fold 0
        score   rmse      time
        train  0.280  7.585205
        valid  0.878  1.508493
        test   0.622  0.268115
CV fold 1
        score   rmse      time
        train  0.317  9.535046
        valid  0.705  1.507614
        test   0.583  0.272807
CV fold 2
        score   rmse       time
        train  0.332  12.457529
        valid  0.709   1.348956
        test   0.563   0.254016
CV fold 3
        score   rmse       time
        train  0.353  16.361878
        valid  0.551   1.354251
        test   0.664   0.256776
CV fold 4
        score   rmse       time
        train  0.360  20.626009
        valid  0.695   1.330484
        test   0.618   0.262307
CV fold 5
        score   rmse       time
        train  0.368  23.560019
        valid  0.706   1.364733
        test   0.615   0.262338
CV fold 6
        score   rmse       time
        train  0.374  29.812642
        valid  0.872   1.364596
        test   0.606   0.264050
CV fold 7
        score   rmse       time
        train  0.374  28.971641
        valid  0.684   1.429442
        test   0.721   0.258017
----------------------------
Mean CV scores
----------------------------
        score            rmse         time
        train  0.345 ± 0.0309  18.6 ± 7.97
        valid  0.725 ± 0.0993   1.4 ± 0.07
        test    0.624 ± 0.046   0.3 ± 0.01
----------------------------
Bagged scores
----------------------------
        score   rmse
        valid  0.732
        test   0.551
```

</details>
