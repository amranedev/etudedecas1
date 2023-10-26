import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import math

# Données de séries temporelles (à remplacer par vos données réelles)
mois = np.arange(1, 54)
prestations = np.array(
    [
        3305345.52,
        2717024.60,
        3100685.38,
        2977755.93,
        2983985.18,
        2927818.68,
        2936467.46,
        2470887.54,
        3007230.13,
        3216363.02,
        2902176.61,
        2882357.68,
        3039295.34,
        2670463.02,
        2979031.61,
        2870829.76,
        2894239.83,
        3006853.57,
        2917417.06,
        2380095.65,
        3138705.23,
        3182097.49,
        2995258.22,
        3005733.84,
        3357566.59,
        3029625.30,
        3316411.76,
        3071064.54,
        3073751.40,
        3345316.27,
        2932189.31,
        2469196.26,
        3118486.28,
        3268170.45,
        3162974.45,
        3153346.19,
        3441295.19,
        3146689.67,
        3561391.80,
        3252127.73,
        3351209.83,
        3439333.42,
        3046435.97,
        2841793.13,
        3396440.55,
        3429367.66,
        3338090.70,
        3293133.07,
        3730948.17,
        3174975.70,
        3479173.23,
        2796642.24,
        2684974.28,
    ]
)

# Divisez les données en ensembles d'entraînement et de test
train_size = int(len(prestations) * 0.8)
train_data = prestations[:train_size]
test_data = prestations[train_size:]

# Modèle de régression linéaire
regression_model = LinearRegression()
regression_model.fit(mois[:train_size].reshape(-1, 1), train_data)
regression_predictions = regression_model.predict(mois[train_size:].reshape(-1, 1))

# Métriques d'évaluation pour la régression linéaire
regression_rmse = math.sqrt(mean_squared_error(test_data, regression_predictions))
regression_mae = mean_absolute_error(test_data, regression_predictions)
regression_r2 = r2_score(test_data, regression_predictions)

print("Régression Linéaire - RMSE:", regression_rmse)
print("Régression Linéaire - MAE:", regression_mae)
print("Régression Linéaire - R2:", regression_r2)

# Modèle ARIMA
order_arima = (1, 1, 1)  # Remplacez par les ordres appropriés
model_arima = ARIMA(train_data, order=order_arima)
result_arima = model_arima.fit()
arima_predictions = result_arima.forecast(steps=len(test_data))

# Métriques d'évaluation pour ARIMA
arima_rmse = math.sqrt(mean_squared_error(test_data, arima_predictions))
arima_mae = mean_absolute_error(test_data, arima_predictions)
arima_r2 = r2_score(test_data, arima_predictions)

print("ARIMA - RMSE:", arima_rmse)
print("ARIMA - MAE:", arima_mae)
print("ARIMA - R2:", arima_r2)

# Modèle SARIMA
order_sarima = (1, 1, 1)  # Remplacez par les ordres appropriés
seasonal_order_sarima = (1, 1, 1, 12)  # Remplacez par les ordres saisonniers appropriés
model_sarima = SARIMAX(
    train_data, order=order_sarima, seasonal_order=seasonal_order_sarima
)
result_sarima = model_sarima.fit()
sarima_predictions = result_sarima.get_forecast(steps=len(test_data))

# Métriques d'évaluation pour SARIMA
sarima_rmse = math.sqrt(
    mean_squared_error(test_data, sarima_predictions.predicted_mean)
)
sarima_mae = mean_absolute_error(test_data, sarima_predictions.predicted_mean)
sarima_r2 = r2_score(test_data, sarima_predictions.predicted_mean)

print("SARIMA - RMSE:", sarima_rmse)
print("SARIMA - MAE:", sarima_mae)
print("SARIMA - R2:", sarima_r2)

# Modèle Simple Exponential Smoothing (SES)
model_ses = SimpleExpSmoothing(train_data)
model_ses_fit = model_ses.fit()
ses_predictions = model_ses_fit.forecast(steps=len(test_data))

# Métriques d'évaluation pour SES
ses_rmse = math.sqrt(mean_squared_error(test_data, ses_predictions))
ses_mae = mean_absolute_error(test_data, ses_predictions)
ses_r2 = r2_score(test_data, ses_predictions)

print("SES - RMSE:", ses_rmse)
print("SES - MAE:", ses_mae)
print("SES - R2:", ses_r2)

# Modèle Holt-Winters (Triple Exponential Smoothing)
holt_winters_model = ExponentialSmoothing(
    train_data, seasonal="add", seasonal_periods=6
)
holt_winters_fit = holt_winters_model.fit()
holt_winters_predictions = holt_winters_fit.forecast(steps=len(test_data))

# Métriques d'évaluation pour Holt-Winters
holt_winters_rmse = math.sqrt(mean_squared_error(test_data, holt_winters_predictions))
holt_winters_mae = mean_absolute_error(test_data, holt_winters_predictions)
holt_winters_r2 = r2_score(test_data, holt_winters_predictions)

print("Holt-Winters - RMSE:", holt_winters_rmse)
print("Holt-Winters - MAE:", holt_winters_mae)
print("Holt-Winters - R2:", holt_winters_r2)


# Modèle LSTM
def prepare_data(series, time_steps):
    X, y = [], []
    for i in range(len(series) - time_steps):
        X.append(series[i : i + time_steps])
        y.append(series[i + time_steps])
    return np.array(X), np.array(y)


time_steps = 12
X, y = prepare_data(train_data, time_steps)
X = X.reshape(X.shape[0], time_steps, 1)

model_lstm = Sequential()
model_lstm.add(LSTM(50, activation="relu", input_shape=(time_steps, 1)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer="adam", loss="mse")
model_lstm.fit(X, y, epochs=100, batch_size=32, verbose=0)

lstm_predictions = []

for i in range(len(test_data)):
    if i < time_steps:
        X_test = np.array(train_data[-time_steps:]).reshape(1, time_steps, 1)
    else:
        X_test = np.array(lstm_predictions[-time_steps:]).reshape(1, time_steps, 1)

    prediction = model_lstm.predict(X_test)
    lstm_predictions.append(prediction[0, 0])

lstm_rmse = math.sqrt(mean_squared_error(test_data, lstm_predictions))
lstm_mae = mean_absolute_error(test_data, lstm_predictions)
lstm_r2 = r2_score(test_data, lstm_predictions)

print("LSTM - RMSE:", lstm_rmse)
print("LSTM - MAE:", lstm_mae)
print("LSTM - R2:", lstm_r2)

# Comparaison des modèles
models = ["Régression Linéaire", "ARIMA", "SARIMA", "SES", "Holt-Winters", "LSTM"]
rmse_scores = [
    regression_rmse,
    arima_rmse,
    sarima_rmse,
    ses_rmse,
    holt_winters_rmse,
    lstm_rmse,
]
mae_scores = [
    regression_mae,
    arima_mae,
    sarima_mae,
    ses_mae,
    holt_winters_mae,
    lstm_mae,
]
r2_scores = [regression_r2, arima_r2, sarima_r2, ses_r2, holt_winters_r2, lstm_r2]

# Affichage des résultats
results_df = pd.DataFrame(
    {"Modèle": models, "RMSE": rmse_scores, "MAE": mae_scores, "R2": r2_scores}
)
print(results_df)

# Choix du meilleur modèle (par exemple, en minimisant RMSE ou MAE, ou en maximisant R2)
best_model = results_df.loc[results_df["RMSE"].idxmin()]
print("Meilleur modèle (selon RMSE):\n", best_model)
