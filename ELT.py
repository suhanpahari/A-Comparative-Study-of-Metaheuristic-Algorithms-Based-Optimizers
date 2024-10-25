# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:09:24 2024

@author: pahar
"""

N_JOBS = -1

# Random variable for having consistent results between runs
RANDOM_STATE = 18

# Dataset's path location
DATASET_SRC = 'C:/Users/pahar/.cache/kagglehub/datasets/abhisheksjha/time-series-air-quality-data-of-india-2010-2023/versions/2'

df_states = pd.read_csv(f'{DATASET_SRC}/stations_info.csv')
df_states.drop(columns=['agency', 'station_location', 'start_month'], inplace=True)
df_states.head()

def combine_state_df(state_name):
    '''
    Combine all state files into a single dataframe and attach the city information.

    Parameters
    ----------
        state_name (str): The name of the state

    Return
    ------
        df (DataFrame): The combined dataframe from all files of a specific state
    '''

    state_code = df_states[df_states['state'] == state_name]['file_name'].iloc[0][:2]
    state_files = glob.glob(f'{DATASET_SRC}/{state_code}*.csv')
    print(f'Combining a total of {len(state_files)} files...\n')

    combined_df = []

    for state_file in state_files:
        # Use os.path.basename to extract the file name without relying on path structure
        file_name = os.path.basename(state_file)[0:-4]  # Removes .csv extension

        # Read CSV file into DataFrame
        file_df = pd.read_csv(state_file)

        # Safely extract the city from df_states and attach to file_df
        city_info = df_states[df_states['file_name'] == file_name]['city'].values
        if len(city_info) > 0:
            file_df['city'] = city_info[0]
        else:
            print(f'City information not found for file: {file_name}')
            file_df['city'] = 'Unknown'

        # Ensure the 'city' column is of type string
        file_df['city'] = file_df['city'].astype('string')

        # Append the DataFrame to the list
        combined_df.append(file_df)

    # Return the concatenated DataFrame
    return pd.concat(combined_df, ignore_index=True)

df = combine_state_df('Delhi')
df.info()

# Make the 'From Date' column the index as datetime
def create_dt_index(dataframe):
    dataframe = dataframe.drop(columns='To Date')
    dataframe['From Date'] = pd.to_datetime(dataframe['From Date'])
    dataframe = dataframe.rename(columns={'From Date': 'datetime'})
    return dataframe.set_index('datetime')

df = create_dt_index(df)
df.head(2)

"""### Cleaning"""

columns_to_keep = ['PM2.5 (ug/m3)', 'CO (mg/m3)', 'Ozone (ug/m3)', 'NOx (ug/m3)']

df = df[columns_to_keep]

def remove_outliers(df):

    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove outliers
    df_no_outliers = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

    return df_no_outliers

def mean_by_index_with_outlier_removal(df):

    # Remove outliers
    df_cleaned = remove_outliers(df)

    # Group by index and calculate the mean
    df_mean = df_cleaned.groupby(df_cleaned.index).mean()

    return df_mean

df_mean = mean_by_index_with_outlier_removal(df)

df = df_mean

df.index.duplicated().any()

df = df.interpolate(method='pad')
df = df.fillna(df.mean())
df.info()

df.describe()

"""### Analyzation And Transformation"""

df.index = pd.to_datetime(df.index)

columns_to_avg = ['PM2.5 (ug/m3)', 'CO (mg/m3)', 'Ozone (ug/m3)', 'NOx (ug/m3)']


daily_avg_df = pd.DataFrame()


start_date = '2016-01-01'
end_date = '2023-03-31'

date_range = pd.date_range(start=start_date, end=end_date, freq='D')


for date in date_range:
    daily_avg = df.loc[date.strftime('%Y-%m-%d'), columns_to_avg].max()
    daily_avg = pd.DataFrame(daily_avg).T
    daily_avg['Date'] = date
    daily_avg_df = pd.concat([daily_avg_df, daily_avg], ignore_index=True)


daily_avg_df.set_index('Date', inplace=True)

dfd = daily_avg_df

dfw = dfd.resample('W').mean()

dfw.head()