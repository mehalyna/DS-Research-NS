def calculate_caffeine_per_cup(coffee_intake, caffeine_mg):
    """
    Calculates the caffeine content per cup based on total intake.
    Returns 0.0 if intake is zero to prevent division errors.
    """
    try:
        coffee = float(coffee_intake)
        caffeine = float(caffeine_mg)
    except (ValueError, TypeError):
        return 0.0

    return caffeine / coffee if coffee > 0 else 0.0

def add_derived_features(user_data):
    """
    Applies standard feature engineering to raw user data.
    Returns a copy of the data with new features added.
    """
    enriched_data = user_data.copy() 
    
    enriched_data['Caffeine_per_Cup'] = calculate_caffeine_per_cup(
        enriched_data.get('Coffee_Intake', 0),
        enriched_data.get('Caffeine_mg', 0)
    )
    
    return enriched_data