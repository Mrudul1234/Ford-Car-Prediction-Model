# Ford Car Price Prediction

![Ford Mustang](ford.jpg)

A Linear Regression model that predicts the resale price of a used Ford car from its year, mileage, engine size, fuel type, transmission and model — built as an end-to-end ML project to actually understand what happens between "load the CSV" and "get a number out."

## The data

`ford.csv` — 17,966 used Ford listings from the [Ford Car Price Prediction](https://www.kaggle.com/datasets/adhurimquku/ford-car-price-prediction) on Kaggle. Each row is one listing:

| column | meaning |
|---|---|
| model | Ford model (Fiesta, Focus, Mustang, Kuga, etc.) |
| year | registration year |
| price | sale price (target) |
| transmission | Manual / Automatic / Semi-Auto |
| mileage | odometer reading |
| fuelType | Petrol / Diesel / Hybrid / Electric / Other |
| tax | road tax |
| mpg | miles per gallon |
| engineSize | engine size in litres |

`test.csv` is the held-out test split after encoding and scaling — this is what the notebook's evaluation and manual prediction cells actually run against.

## What's in the notebook (`notebook.ipynb`)

1. **EDA** — correlation heatmap, scatter/bar plots of price against year, model, fuel type, mileage, before deciding anything about the model.
2. **Encoding** — one-hot encoded `model`, `fuelType`, `transmission`, since Linear Regression needs numbers, not text categories.
3. **Feature selection** — Pearson correlation of every feature against price, plus chi-square tests on the categorical ones (price binned into quartiles for the chi-square step), to drop columns with no real relationship to price instead of throwing everything at the model.
4. **Scaling** — numeric columns (`year`, `mileage`, `tax`, `mpg`, `engineSize`) scaled before fitting.
5. **Model** — plain Linear Regression on the selected features.
6. **Evaluation** — on the held-out test set:

   | metric | value |
   |---|---|
   | R² | 0.85 |
   | MAE | $1,370 |
   | RMSE | $1,835 |

   So the model explains about 85% of the variance in price, and its predictions are off by roughly $1,370 on average.

7. **Manual spot-check** — ran the trained model against 11 real listings I picked by hand and compared actual vs. predicted price for each; average error there was ~$1,477, close to the formal test MAE.

## Correlation highlights

From the Pearson step, the strongest single predictors of price were:

- `year` → +0.64 (newer cars sell for more)
- `mileage` → −0.53 (higher mileage, lower price)
- `engineSize` → +0.50 (bigger engine, higher price)

Which lines up with how you'd expect used car pricing to work — good to see it actually show up in the numbers instead of just assuming it.

## Files

- `ml2.ipynb` — the full notebook: EDA → encoding → feature selection → model → evaluation
- `ford.csv` — raw dataset
- `test.csv` — processed/scaled test split
- `ford.jpg` — banner image

## What I'd add next

- Compare against Random Forest / Gradient Boosting to see how much a non-linear model actually gains over plain Linear Regression
- Cross-validation instead of a single train/test split, to check how stable the R² actually is
- Residual plots to check where the model is systematically off (likely on higher-end/rare models with few listings)

This is a learning project — first full ML pipeline I've built end to end, feedback welcome.
