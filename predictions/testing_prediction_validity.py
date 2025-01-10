import pandas as pd
import pickle
from predictions.make_predictions import predict_habitability
from tensorflow.keras.models import load_model


"""MAKE PREDICTIONS - TESTING MODEL VALIDITY"""

def test_model_validity():

  # load the ML model and Scaler
  model = load_model('./saved_models/exoplanet_habitat_model.keras')
  with open('./saved_models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


  # normal earth data
  normal_earth_statement = predict_habitability([1, 1, 1, 365.00000, 1.0, 1.0, 1.0000, 0.0034, 255.0, 5778.0, 1.00, 1.00])

  # already scaled earth data
  already_scaled_earth_statement = predict_habitability([0.000000, 0.000000, 1.0, 9.077346e-07, 0.000182, 0.000037, 0.000039, 0.003505, 0.066586, 0.094778, 0.011191, 0.101010])

  # normal earth data, when scaled with the scaler and predicted manually without the function
  normal_earth = pd.DataFrame({
      'num_stars': [1],
      'num_planets': [1],
      'num_moons': [1],
      'orbital_period': [365],
      'radius_in_earth': [1],
      'mass_in_earth': [1],
      'planet_mass': [1],
      'eccentricity': [0.0034],
      'equilibrium_temp_K': [255],
      'st_eff_temp_k': [5778],
      'st_radius': [1],
      'st_mass': [1]
  })
  normal_earth_scaled = scaler.transform(normal_earth)
  print(normal_earth_scaled) # should match the already scaled earth data array
  predicted_score_normal_earth = model.predict(normal_earth_scaled)

  # print out the results
  print("\n\nNormal Earth: " + normal_earth_statement)
  print("Already Scaled Earth: " + already_scaled_earth_statement)

  # should match the prediction results when normal earth data is passed into predict_habitability
  print(f"Manually Scaled + Predicted Earth: Predicted Habitability Score: {predicted_score_normal_earth[0][0]:.4f}\n\n\n")
