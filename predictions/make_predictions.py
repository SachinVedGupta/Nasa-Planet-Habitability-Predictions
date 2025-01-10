import pandas as pd
from tensorflow.keras.models import load_model
import pickle

"""MAKE PREDICTIONS - FUNCTIONS"""

# turn user input into the formatted array for a planet's data
def input_to_planet_list():
    print("Enter the following details for the exoplanet:")

    num_stars = float(input("Number of Stars (num_stars): "))
    num_planets = float(input("Number of Planets (num_planets): "))
    num_moons = float(input("Number of Moons (num_moons): "))
    orbital_period = float(input("Orbital Period (orbital_period in years): "))
    radius_in_earth = float(input("Radius in Earth (radius_in_earth): "))
    mass_in_earth = float(input("Mass in Earth (mass_in_earth): "))
    planet_mass = float(input("Planet Mass (planet_mass): "))
    eccentricity = float(input("Eccentricity (eccentricity): "))
    equilibrium_temp_K = float(input("Equilibrium Temperature (equilibrium_temp_K in Kelvin): "))
    st_eff_temp_k = float(input("Star Effective Temperature (st_eff_temp_k in Kelvin): "))
    st_radius = float(input("Star Radius (st_radius in solar radii): "))
    st_mass = float(input("Star Mass (st_mass in solar masses): "))

    return [num_stars, num_planets, num_moons, orbital_period, radius_in_earth, mass_in_earth, planet_mass, eccentricity, equilibrium_temp_K, st_eff_temp_k, st_radius, st_mass]


# use the model to predict the planet's habitability
def predict_habitability(planet_list, planet_name=""):
    
    # load the ML model and Scaler
    model = load_model('./saved_models/exoplanet_habitat_model.keras')
    with open('./saved_models/scaler.pkl', 'rb') as f:
      scaler = pickle.load(f)

    # format of planet_list = [num_stars, num_planets, num_moons, orbital_period, radius_in_earth, mass_in_earth, planet_mass, eccentricity, equilibrium_temp_K, st_eff_temp_k, st_radius, st_mass]
    # print(planet_list)

    # Create DataFrame for the new input
    new_exoplanet = pd.DataFrame({
        'num_stars': [planet_list[0]],
        'num_planets': [planet_list[1]],
        'num_moons': [planet_list[2]],
        'orbital_period': [planet_list[3]],
        'radius_in_earth': [planet_list[4]],
        'mass_in_earth': [planet_list[5]],
        'planet_mass': [planet_list[6]],
        'eccentricity': [planet_list[7]],
        'equilibrium_temp_K': [planet_list[8]],
        'st_eff_temp_k': [planet_list[9]],
        'st_radius': [planet_list[10]],
        'st_mass': [planet_list[11]]
    })

    # Scale the new data
    new_exoplanet_scaled = scaler.transform(new_exoplanet)

    # Make the prediction
    predicted_score = model.predict(new_exoplanet_scaled)

    # Display the predicted habitability score
    add_planet_name = ""
    if planet_name != "":
      add_planet_name = f" for \n   {planet_name.title()}"

    test_scaling = False
    if test_scaling:
      print(f"Normal DF:", new_exoplanet)
      print(f"Scaled DF:", new_exoplanet_scaled)

    return f"   Predicted Habitability Score{add_planet_name}: {predicted_score[0][0]:.4f}"


# Call the function to predict habitability
    # predict_habitability(input_to_planet_list())