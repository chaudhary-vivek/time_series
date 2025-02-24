import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
    

    

def data_load(PATH , FORECAST_DATE, DATE_COL, Y_VAR):
    df = pd.read_csv('Walmart.csv')
    # df = pd.read_csv(f'{PATH}/Walmart.csv')
    df[DATE_COL] = pd.to_datetime(df[DATE_COL] , format='%d-%m-%Y')
    df[Y_VAR] = df[Y_VAR].astype(float)
    df = df.sort_values(by = DATE_COL)
    train = df[df[DATE_COL] <= FORECAST_DATE]
    test = df[df[DATE_COL] > FORECAST_DATE]
    test = test.rename(columns = {Y_VAR : Y_VAR + '_test'})

    return train, test

def smape(actual, predicted):
    numerator = np.abs(actual - predicted)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return 100 * np.mean(numerator / denominator)


def get_accuracy(forecast_df, test_df, KEY_COL, DATE_COL, Y_VAR):
    df = forecast_df.merge(test_df, on = [KEY_COL, DATE_COL], how = 'left')
    smape_by_group = df.groupby(KEY_COL).apply(
    lambda x: smape(x[Y_VAR], x[Y_VAR + '_test'])).reset_index()
    smape_by_group.columns = [KEY_COL, 'SMAPE']
    return smape_by_group



def plot_accuracy(forecast_df, test_df, KEY_COL, DATE_COL, Y_VAR, chart_title):
    df = get_accuracy(forecast_df, test_df, KEY_COL, DATE_COL, Y_VAR)

    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df['SMAPE'])

    median = df['SMAPE'].median()
    plt.axhline(y=median, color='red', linestyle='--')
    plt.text(0.1, median, f'Median: {median:.2f}%', 
            verticalalignment='bottom')

    plt.title(f'Distribution of SMAPE Values for {chart_title}')
    plt.ylabel('SMAPE (%)')
    plt.grid(True)
    plt.show()




def plot_comparison(forecast_df, test_df, n_graphs, KEY_COL, DATE_COL, Y_VAR):
    df_ = forecast_df.merge(test_df, on = [KEY_COL, DATE_COL], how = 'left')

    for key in df_[KEY_COL].unique()[0:n_graphs]:
        df = df_[df_[KEY_COL] == key]
        plt.figure(figsize=(15, 6))
        plt.plot(df[DATE_COL], df[Y_VAR], label= 'forecast', color='blue')
        plt.plot(df[DATE_COL], df[Y_VAR + '_test'], label='actual', color='red')
        plt.title(key)

def train_model(train, method, KEY_COL ,DATE_COL, Y_VAR, FORECAST_PERIODS):
    all_forecasts = pd.DataFrame()
    for key in train[KEY_COL].unique():
        df = train[train[KEY_COL] == key]
        df = df[[KEY_COL ,DATE_COL, Y_VAR]]

        if method == 'arima':
            forecasts = forecast_arima(df,  DATE_COL, Y_VAR,  FORECAST_PERIODS, KEY_COL)
        elif method == 'lstm':
            forecasts = forecast_lstm(df,  DATE_COL, Y_VAR,  FORECAST_PERIODS, KEY_COL)

        all_forecasts = pd.concat([all_forecasts, forecasts])

    return all_forecasts
    


def forecast_arima(df, DATE_COL, Y_VAR, FORECAST_PERIODS, KEY_COL):

    order=(1,1,1)

    key = df[KEY_COL].unique()[0]
    
    model = ARIMA(df[Y_VAR].values, order=order)
    model_fit = model.fit()
    
    last_date = df[DATE_COL].iloc[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=FORECAST_PERIODS,
        freq=pd.infer_freq(df[DATE_COL])
    )
    
    forecast = model_fit.forecast(steps=FORECAST_PERIODS)
    
    forecast_df = pd.DataFrame({
        KEY_COL : key,
        DATE_COL: forecast_dates,
        Y_VAR: forecast,
        'is_forecast': True
    })
    
    df['is_forecast'] = False
    
    result = pd.concat([df, forecast_df], axis=0)
    result['Model'] = 'ARIMA'
    
    return result


def forecast_lstm(df, DATE_COL, Y_VAR, FORECAST_PERIODS, KEY_COL):

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    
    key = df[KEY_COL].unique()[0]
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[Y_VAR].values.reshape(-1, 1))
    
    seq_length = 10
    X, y = create_sequences(scaled_data, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(X, y, epochs=50, batch_size=32, verbose=0)
    
    last_sequence = scaled_data[-seq_length:]
    forecast_scaled = []
    
    for _ in range(FORECAST_PERIODS):
        current_sequence = last_sequence.reshape(1, seq_length, 1)
        next_pred = model.predict(current_sequence, verbose=0)
        forecast_scaled.append(next_pred[0, 0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_pred
    
    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1))
    
    last_date = df[DATE_COL].iloc[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=FORECAST_PERIODS,
        freq=pd.infer_freq(df[DATE_COL])
    )
    
    forecast_df = pd.DataFrame({
        KEY_COL: key,
        DATE_COL: forecast_dates,
        Y_VAR: forecast.flatten(),
        'is_forecast': True
    })
    
    df['is_forecast'] = False
    
    result = pd.concat([df, forecast_df], axis=0)
    result['Model'] = 'LSTM'
    
    return result