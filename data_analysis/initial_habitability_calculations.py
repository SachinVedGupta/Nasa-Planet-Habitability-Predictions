import numpy as np

# Functions to calculate habitability score based on closeness to Earth's statistics

def calculate_habitability_score(row, earth_stats):
    score = 0.0

    # Calculate individual scores based on proximity to Earth's values
    score += 1 - abs(row['num_stars'] - earth_stats['num_stars'])
    score += 1 - abs(row['num_planets'] - earth_stats['num_planets'])
    score += 1 - abs(row['num_moons'] - earth_stats['num_moons'])
    score += 1 - abs(row['orbital_period'] - earth_stats['orbital_period'])
    score += 1 - abs(row['radius_in_earth'] / earth_stats['radius_in_earth'])  # Normalized to Earth's radius
    score += 1 - abs(row['mass_in_earth'] / earth_stats['mass_in_earth'])  # Normalized to Earth's mass
    score += 1 - abs(row['eccentricity'] - earth_stats['eccentricity'])
    score += 1 - abs(row['equilibrium_temp_K'] - earth_stats['equilibrium_temp_K'])
    score += 1 - abs(row['st_eff_temp_k'] - earth_stats['st_eff_temp_k'])
    score += 1 - abs(row['st_radius'] - earth_stats['st_radius'])
    score += 1 - abs(row['st_mass'] - earth_stats['st_mass'])

    # Average the score (divided by the number of features)
    normalized_score = score / len(earth_stats)

    return min(max(normalized_score, 0), 1)  # Scale score between 0 and 1



def get_closeness(a, b, epsilon=1e-9):
    max_val = max(abs(a), abs(b))
    closeness = 1 - (abs(a - b) / (max_val + epsilon))
    return closeness  # Rounded to 1 decimal place

def calculate_habitability_score_2(row, earth_stats):
    score = 0.0

    # Calculate individual scores based on proximity to Earth's values
    score += 1 - abs(row['num_stars'] - earth_stats['num_stars'])
    score += 1 - abs(row['num_planets'] - earth_stats['num_planets'])
    score += 1 - abs(row['num_moons'] - earth_stats['num_moons'])
    score += 1 - abs(row['orbital_period'] - earth_stats['orbital_period'])
    score += get_closeness(row['radius_in_earth'], earth_stats['radius_in_earth']) # Normalized to Earth's radius
    score += get_closeness(row['mass_in_earth'], earth_stats['mass_in_earth']) # Normalized to Earth's radius
    score += 1 - abs(row['eccentricity'] - earth_stats['eccentricity'])
    score += 1 - abs(row['equilibrium_temp_K'] - earth_stats['equilibrium_temp_K'])
    score += 1 - abs(row['st_eff_temp_k'] - earth_stats['st_eff_temp_k'])
    score += 1 - abs(row['st_radius'] - earth_stats['st_radius'])
    score += 1 - abs(row['st_mass'] - earth_stats['st_mass'])

    # Average the score (divided by the number of features)
    normalized_score = score / 11.0

    return normalized_score  # Scale score between 0 and 1



def calculate_habitability_score_3(row, earth_stats, weights=[0.1, 0.1, 0.05, 0.1, 0.15, 0.15, 0.05, 0.1, 0.1, 0.05, 0.05]):
    # Normalize each feature relative to Earth's stats
    def normalized_similarity(value, earth_value, scale=1.0):
        return 1 - (abs(value - earth_value) / scale)

    # Define feature importance weights
    feature_weights = np.array(weights)

    # Calculate normalized similarities for each feature
    similarities = np.array([
        normalized_similarity(row['num_stars'], earth_stats['num_stars'], scale=2.0),
        normalized_similarity(row['num_planets'], earth_stats['num_planets'], scale=5.0),
        normalized_similarity(row['num_moons'], earth_stats['num_moons'], scale=10.0),
        normalized_similarity(row['orbital_period'], earth_stats['orbital_period'], scale=1000.0),
        normalized_similarity(row['radius_in_earth'], earth_stats['radius_in_earth'], scale=10.0),
        normalized_similarity(row['mass_in_earth'], earth_stats['mass_in_earth'], scale=10.0),
        normalized_similarity(row['eccentricity'], earth_stats['eccentricity'], scale=1.0),
        normalized_similarity(row['equilibrium_temp_K'], earth_stats['equilibrium_temp_K'], scale=1000.0),
        normalized_similarity(row['st_eff_temp_k'], earth_stats['st_eff_temp_k'], scale=1000.0),
        normalized_similarity(row['st_radius'], earth_stats['st_radius'], scale=5.0),
        normalized_similarity(row['st_mass'], earth_stats['st_mass'], scale=5.0)
    ])

    # Weighted average of similarities
    weighted_score = np.sum(similarities * feature_weights) / np.sum(feature_weights)

    # Ensure score is between 0 and 1
    return max(0, min(1, weighted_score))