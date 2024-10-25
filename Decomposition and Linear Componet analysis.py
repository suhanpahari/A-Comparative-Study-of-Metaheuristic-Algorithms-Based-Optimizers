# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:10:47 2024

@author: pahar
"""

clm = "PM2.5 (ug/m3)"

plt.figure(figsize=(19,19))

plt.subplot(411)
plt.grid(True)
plt.plot(dfd[clm], label='Original', color = 'Green')
plt.legend(loc='upper left')


def adf_test(series):
    result=adfuller(series)
    print('ADF Statistics: {}'.format(result[0]))
    print('p- value: {}'.format(result[1]))
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")

#checking

adf_test(dfw[clm])

#this function running loop, where it is checking stationarity,and telling us the oder of difference which will make this data staionary
#another worthless code to ease the work, usually we should check mannually one by one


def check_stationarity_and_difference(df,column_name, max_diff=360, significance_level=0.05):

    def adf_test(series):
        result = adfuller(series, autolag='AIC')
        p_value = result[1]
        return p_value

    data = df[column_name].copy()
    diff_count = 0
    p_value = adf_test(data)

    while p_value > significance_level and diff_count < max_diff:
        data = data.diff().dropna()
        diff_count += 1

        new_column_name = f"{column_name} {diff_count}th diff"
        df[new_column_name] = data

        p_value = adf_test(data)
        print(f"Differencing step {diff_count}, p-value: {p_value}")

    if p_value <= significance_level:
        print(f"Data is stationary after {diff_count} differences.")
    else:
        print(f"Data is still non-stationary after {diff_count} differences.")

    return df, diff_count

"""### Decomposition"""


stl = STL(dfw[clm], period = 52)
result = stl.fit()

trend = result.trend
seasonal = result.seasonal
residual = result.resid

plt.figure(figsize=(20,9))

plt.subplot(411)
plt.grid(True)
plt.plot(dfw[clm], label='Original')
plt.legend(loc='upper left')

plt.subplot(412)
plt.grid(True)
plt.plot(trend, label='Trend', color='orange')
plt.legend(loc='upper left')

plt.subplot(413)
plt.grid(True)
plt.plot(seasonal, label='Seasonality', color='green')
plt.legend(loc='upper left')

plt.subplot(414)
plt.grid(True)
plt.plot(residual, label='Residuals', color='red')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()


"""## Trend Analysis"""

trend.tail(5)



train = trend[(trend.index < '2023-03-05')]
test = trend[trend.index>='2023-03-05']

print(train.shape)
print(test.shape)

plt.figure(figsize=(20, 2))

# Plot the train and test data
train.plot(label='Training Data')
test.plot(label='Test Data')

# Adding grid, legend, and title
plt.grid(True)
plt.legend(['Training Data', 'Test Data'])
plt.title(f'{clm} Data Train and Test Split')

# Show the plot
plt.tight_layout()  # Adjust layout to fit everything
plt.show()

# Define the COVID period
covid_start = '2019-06-01'
covid_end = '2021-03-01'

# Step 1: Split the data into pre-COVID and post-COVID
pre_covid_data = train[train.index < covid_start]
post_covid_data = train[train.index > covid_end]

# Step 2: Fit an ARIMA model on pre-COVID data (use your own column, e.g., PM2.5 (ug/m3))
model = ARIMA(pre_covid_data, order=(5, 1, 0))  # Adjust ARIMA (p,d,q) order as needed
model_fit = model.fit()

# Step 3: Generate synthetic data for the COVID period
# Number of points to generate is equal to the length of the missing period
covid_dates = pd.date_range(start=covid_start, end=covid_end, freq='D')
num_points = len(covid_dates)

# Forecasting using the ARIMA model
synthetic_data = model_fit.forecast(steps=num_points)

# Step 4: Create a DataFrame for the synthetic data
synthetic_df = pd.DataFrame({'PM2.5 (ug/m3)': synthetic_data}, index=covid_dates)

# Step 5: Combine synthetic data with original data
combined_data = pd.concat([pre_covid_data, synthetic_df, post_covid_data])

# Step 6: Plot the combined data
plt.figure(figsize=(12, 6))
combined_data.plot(label='Combined Data with Synthetic', color='blue')
train.plot(label='Original Data', color='orange', linestyle='--')
plt.axvspan(covid_start, covid_end, color='red', alpha=0.3, label='Synthetic Data Period')
plt.legend()
plt.grid(True)
plt.title('Synthetic Data Generation for COVID Period')
plt.show()


# Ensure the index is in datetime format if not done already
train.index = pd.to_datetime(train.index)

# Step 1: Filter out the data outside the unwanted date range for fitting
filtered_train = train.loc[(train.index < '2019-06-01') | (train.index > '2021-03-01')]

# Convert the date index to numeric format (ordinal) to use it as X
X_filtered = filtered_train.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1)

# Values as y, access data without column index since no column names are present
y_filtered = filtered_train.values.flatten()  # Flatten to ensure it's a 1D array

# Step 2: Fit linear regression to the filtered data
model = LinearRegression()
model.fit(X_filtered, y_filtered)

# Step 3: Generate predictions for the excluded date range ('2019-06-01' to '2021-03-01')
excluded_dates = pd.date_range('2019-06-01', '2021-03-01', freq='W')  # assuming weekly frequency
X_excluded = excluded_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)

# Predict values for the excluded date range
predicted_values = model.predict(X_excluded)

# Step 4: Replace the data in the original DataFrame for the excluded date range
# Create a DataFrame for the predicted values, ensure it's a 1D array
predicted_df = pd.DataFrame(predicted_values, index=excluded_dates)

# Replace the original data in the specified date range with the predicted values
train.loc[excluded_dates] = predicted_df.values.flatten()  # Flatten to match the single column structure

# Print the updated DataFrame
print(train.loc['2019-05-01':'2021-04-01'])

plt.figure(figsize=(20, 9))

# Plot the train and test data
train.plot(label='Training Data')
test.plot(label='Test Data')

# Adding grid, legend, and title
plt.grid(True)
plt.legend(['Training Data', 'Test Data'])
plt.title(f'{clm} Data Train and Test Split')

# Show the plot
plt.tight_layout()  # Adjust layout to fit everything
plt.show()


def plot_acf_pacf(data, column, lags):

    # Plot ACF
    plt.figure(figsize=(20,9))

    plt.subplot(121)
    plot_acf(data, lags=lags, ax=plt.gca())
    plt.title(f"ACF for {column}")

    # Plot PACF
    plt.subplot(122)
    plot_pacf(data, lags=lags, ax=plt.gca())
    plt.title(f"PACF for {column}")

    plt.tight_layout()
    plt.show()

plot_acf_pacf(train, clm, lags=100)

adf_test(train)


model = ARIMA(train, order=(2, 2, 16))

model_fit = model.fit()

n_forecast = len(test)
predictions = model_fit.forecast(steps=n_forecast)

metrics = evaluate_time_series_metrics(test, predictions)
metrics

metrics = evaluate_time_series_metrics(test, predictions)
metrics

def evaluate_time_series_metrics(actual, predicted):

    # Ensure inputs are numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # Calculate R^2
    r_squared = r2_score(actual, predicted)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(actual, predicted)

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    # Return metrics in a dictionary
    return {
        'RMSE': rmse,
        'R^2': r_squared,
        'MAE': mae,
        'MAPE': mape
    }

print(test)
print(predictions)

plt.figure(figsize=(20,9))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Testing Data')
plt.plot(test.index, predictions, label='Predicted', color='green')
plt.title('Forecasting')
plt.xlabel('Date')
plt.ylabel(clm)

plt.legend(loc = 'best')
plt.grid(True)
plt.show()

"""## Seasonal Analysis"""

train = seasonal[(seasonal.index < '2022-04-01')]
test = seasonal[seasonal.index>='2022-04-01']

print(train.shape)
print(test.shape)

plt.figure(figsize=(20, 5))

# Plot the train and test data
train.plot(label='Training Data')
test.plot(label='Test Data')

# Adding grid, legend, and title
plt.grid(True)
plt.legend(['Training Data', 'Test Data'])
plt.title(f'{clm} Data Train and Test Split')

# Show the plot
plt.tight_layout()  # Adjust layout to fit everything
plt.show()


window_size = 8 # You can adjust the window size to match your needs
smoothed_seasonal = seasonal.rolling(window=window_size, center=True).mean()

# Plot the original seasonal component and the smoothed version
plt.figure(figsize=(20, 5))

# Plot original seasonal component
plt.plot(seasonal.index, seasonal_component, label='Original Seasonal Component', color='orange')

# Plot smoothed seasonal component using Moving Average
plt.plot(seasonal.index, smoothed_seasonal, label=f'Smoothed Seasonal (Moving Avg, {window_size})', color='blue')

# Add grid, labels, and title
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('PM2.5 (ug/m3)')
plt.title('Moving Average on Seasonal Component')
plt.legend()

# Show the plot
plt.show()

train = smoothed_seasonal[(smoothed_seasonal.index < '2022-04-01')]
test = smoothed_seasonal[smoothed_seasonal.index>='2022-04-01']

print(train.shape)
print(test.shape)

plt.figure(figsize=(20, 5))

# Plot the train and test data
train.plot(label='Training Data')
test.plot(label='Test Data')

# Adding grid, legend, and title
plt.grid(True)
plt.legend(['Training Data', 'Test Data'])
plt.title(f'{clm} Data Train and Test Split')

# Show the plot
plt.tight_layout()  # Adjust layout to fit everything
plt.show()

test.dropna(inplace = True)

train.dropna(inplace = True)

adf_test(train)


# Assuming your time series data is in 'train'
# s = 12 for monthly seasonality, P=1, D=1, Q=1

model = SARIMAX(train,
                order=(1, 1, 1),              # (p, d, q)
                seasonal_order=(1, 1, 1, 52))  # (P, D, Q, s)
results = model.fit()

# Forecasting future values
forecast = results.forecast(steps=3)  # Forecast 12 periods ahead
print(forecast)

test.head(3)

n_forecast = 3
predictions = results.forecast(steps=n_forecast)

metrics = evaluate_time_series_metrics(test.head(3), predictions)
metrics