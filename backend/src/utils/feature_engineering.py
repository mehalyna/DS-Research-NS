def calculate_caffeine_per_cup(coffee_intake, caffeine_mg):
    """Calculates caffeine per cup safely to avoid division by zero."""
    # Convert to float just in case the API receives strings
    coffee = float(coffee_intake)
    caffeine = float(caffeine_mg)
    return caffeine / coffee if coffee > 0 else 0.0

def add_derived_features(user_data):
    """Applies all standard feature engineering to raw user data."""
    # Make a copy so we don't accidentally mutate the original request
    enriched_data = user_data.copy() 
    
    enriched_data['Caffeine_per_Cup'] = calculate_caffeine_per_cup(
        enriched_data.get('Coffee_Intake', 0),
        enriched_data.get('Caffeine_mg', 0)
    )
    return enriched_data