Data:
1. Investigate original data source:
    - https://parisdata.opendatasoft.com/explore/dataset/comptage-velo-compteurs/information/?disjunctive.id_compteur&disjunctive.nom_compteur&disjunctive.id&disjunctive.name&disjunctive.channel_id&disjunctive.channel_name&disjunctive.channel_sens&disjunctive.counter
    - https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/information[…]eur&disjunctive.nom_compteur&disjunctive.id&disjunctive.name 
2. Consider adding more data, e.g., 
    - weather data of the time and / or seasons
    - weekends / holidays (school holidays)
    - add specific weekday instead of generalization of the week
    - add specific locational features (e.g. number of supermarkets, number of metro stations, restaurants etc.)
    - public transport strikes
    - schedule of soccer matches / big events
    - number of tourists in town
3. Make some more analysis on the data
    - Drop the year if there is only one?
    - What is the output of the iypnb in specific
4. Standardization of features
    - depending of the used model

Algorithms:
0. baseline
    - Ridge
1. geometric
    - KNN? (with standardization)
2. kernel 
    - SVR (with standardization)
3. tree
    - random forest
    - XGB
    - extratrees?
4. neural networks -> probably overengineering
5. ensemble mehtod? -> stacked generalization