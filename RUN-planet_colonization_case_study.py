from predictions.make_predictions import predict_habitability
from predictions.predictions_with_images import predict_habitability_with_visual
from create_ML_model.train_ml_model import create_planet_prediction_model

''' RUN THIS PYTHON FILE TO VIEW THE CASE STUDY '''

''' OPTIONALLY ADJUST THE HYPER-PARAMETERS BELOW '''
reTrain = True
showVisual = True


def run_case_study():

  if reTrain:
    create_planet_prediction_model()

  # format of planet_list = [num_stars, num_planets, num_moons, orbital_period, radius_in_earth, mass_in_earth, planet_mass, eccentricity, equilibrium_temp_K, st_eff_temp_k, st_radius, st_mass]
  earth_mass = 5.972 * 10**24

  earth_reference = [1, 1, 1, 365, 1, 1, 1, 0.034, 255, 5778, 1, 1]

  mars = [1, 1, 2, 687, 0.532, 0.107, (0.641 * 10**24) / earth_mass, 0.093, 210, 5778, 1, 1]
  the_moon = [1, 1, 0, 27.3, 0.273, 0.0123, (7.34 * 10**22) / earth_mass, 0.0549, 250, 5778, 1, 1]
  europa = [1, 1, 0, 3.55, 0.245, 0.008, (4.8 * 10**22) / earth_mass, 0.009, 103, 5778, 1, 1]
  kepler_442b = [1, 1, 0, 112, 1.34, 2, (8 * 10**24) / earth_mass, 0.04, 233, 4402, 0.6, 0.66]
  k2_18b = [1, 1, 0, 33, 2.61, 8.63, (22.5 * 10**24) / earth_mass, 0.23, 282, 3500, 0.4, 0.36]

  # information for getting the predictions
  get_predictions_list = [[earth_reference, "Earth for Reference"], [mars, "Mars"], [the_moon, "Earth's Moon"], [europa, "Europa"], [kepler_442b, "Kepler-442b"], [k2_18b, "K2-18b"]]

  # predicting the habitability scores (higher habitability score means better for colonization/living) with/without the planet visuals
  
  prediction_statements = []
  for item in get_predictions_list:
    if showVisual:
      prediction_statements.append(predict_habitability_with_visual(item[0], item[1]))
    else:
      prediction_statements.append(predict_habitability(item[0], item[1]))


  # output the case study

  print(
  """\n\n\n
##########################################################################################

  CASE STUDY: DETERMINE THE BEST CANDIDATE (PLANET) FOR COLONIZATION

  
  According to NASA, there are 5 main candidates for potential colonization:

  1. Mars
  2. The Moon
  3. Europa (one of Jupiter's moons)
  4. Kepler-442b
  5. K2-18b

  
  Below these planets will be tested via the ML model to determine the best candidate for colonization:
  """)


  for statement in prediction_statements:
    print(statement + "\n")

  print(
  """
  Therefore, based on the results, Mars is the best planet to colonize.

  This is largerly because Mars is in the same universe as Earth, so many characteristics are the same.
  Especially characteristics related to stars, since they share the same star of the Sun.
  This also goes for The Moon (Earth's Moon) and Europa (one of Jupiter's Moons).

  Kepler-442b and K2-18b are exoplanets, meaning they are not in our (Earth's) solar system
  so will more varying attributes as they don't share the the same star(s). 
  Kepler-442b has a predicted habitability score significantly higher than K2-18b, 
  meaning its the best exoplanet option. It ended up having a higher habitability score than the moon, 
  which validates it as a candidate, since it shows that many of its overall attributes match Earth, 
  compared to just relying on the alike Star related attributes to prop up the habitability score 
  (of which applies to planets in our solar system).

##########################################################################################\n\n\n
  """)

# to run/see the case_study
run_case_study()

# to run testing function:
#   from predictions.testing_prediction_validity import test_model_validity
#   test_model_validity()