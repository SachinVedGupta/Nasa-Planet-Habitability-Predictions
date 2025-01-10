from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from data_analysis.planet_data_analysis import get_initial_dataframe

def create_planet_prediction_model():
  # Load the dataset
  data = get_initial_dataframe()

  # Separate features (inputs) and targets (outputs)
  X = data[['num_stars', 'num_planets', 'num_moons', 'orbital_period',
            'radius_in_earth', 'mass_in_earth', 'planet_mass',
            'eccentricity', 'equilibrium_temp_K',
            'st_eff_temp_k', 'st_radius', 'st_mass']]
  
  y = data['habitability_score']

  # Split the data into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Define the model
  model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')  # For regression output
  ])

  # Compile the model
  model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

  # Train the model
  history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2)

  # Evaluate the model on test data
  loss, mae = model.evaluate(X_test, y_test)
  print(f"Test Mean Absolute Error: {mae:.4f}")

  # Make predictions
  predictions = model.predict(X_test)

  # visualization of the model's results
  import matplotlib.pyplot as plt

  plt.scatter(y_test, predictions)
  plt.xlabel('True Habitability Score')
  plt.ylabel('Predicted Habitability Score')
  plt.title('True vs Predicted Habitability Scores')
  plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Diagonal line
  plt.show()

  # Save the model - USE .keras NOT .h5
  model.save('./saved_models/exoplanet_habitat_model.keras')

