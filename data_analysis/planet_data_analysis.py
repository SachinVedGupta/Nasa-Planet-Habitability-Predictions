# import libraries
import pandas as pd
import numpy as np
import pickle
from data_analysis.initial_habitability_calculations import calculate_habitability_score_3

def get_initial_dataframe():

  """Import the Data"""

  df = pd.read_csv('./data_analysis/planet_data.csv')
  df.head()


  """Clean the data"""

  #import skikit learn
  from sklearn.preprocessing import MinMaxScaler, StandardScaler

  #remove row id column
  df = df.drop('row_id', axis=1)

  #show data
  df.head()

  # Initialize the scaler
  scaler = MinMaxScaler()  # Use StandardScaler() for standard scaling

  # Select numeric columns only
  numeric_df = df.select_dtypes(include=['float64', 'int64'])

  #Fill NaN values with the mean of each column
  numeric_df.fillna(numeric_df.mean(), inplace=True)

  """ SCALE THE DATA"""

  # Scale the numeric columns
  scaled_data = scaler.fit_transform(numeric_df)

  # save the scaler
  with open('./saved_models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

  # Convert back to DataFrame
  df_scaled = pd.DataFrame(scaled_data, columns=numeric_df.columns)

  # Replace original numeric columns in the original DataFrame
  df[df.columns.intersection(numeric_df.columns)] = df_scaled

  # show data
  df.head()

  # show data shape
  df.shape

  """ APPLY HABITABILITY SCORES"""

  # Define Earth's scaled statistics
  earth_stats = df.iloc[0]

  # Calculate habitability scores and assign them to a new column in the DataFrame
  df['habitability_score_similar'] = df.apply(calculate_habitability_score_3, axis=1, args=(earth_stats,))
  df['habitability_score'] = (df['habitability_score_similar'] - df['habitability_score_similar'].min()) / (df['habitability_score_similar'].max() - df['habitability_score_similar'].min())

  # print(df)
  return df