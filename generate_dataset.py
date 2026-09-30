import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_crime_data(num_records=12000):
    np.random.seed(42)
    random.seed(42)

    # Delhi center coordinates
    base_lat = 28.7041
    base_lon = 77.1025

    # Crime Types, Weapons, and locations
    crime_types = ['Theft', 'Robbery', 'Burglary', 'Assault', 'Murder', 'Kidnapping', 'Fraud', 'Cybercrime', 'Vandalism']
    weapons = ['None', 'Knife', 'Firearm', 'Blunt Object', 'Poison', 'Rope']
    
    # Generate some hotspots (centers for clustering)
    hotspots = [
        (base_lat + 0.05, base_lon + 0.05),
        (base_lat - 0.06, base_lon - 0.04),
        (base_lat + 0.08, base_lon - 0.07),
        (base_lat - 0.03, base_lon + 0.08),
        (base_lat, base_lon)
    ]

    records = []
    start_date = datetime(2023, 1, 1)
    
    for _ in range(num_records):
        # 70% chance to be near a hotspot, 30% random
        if random.random() < 0.7:
            hx, hy = random.choice(hotspots)
            lat = np.random.normal(hx, 0.01)
            lon = np.random.normal(hy, 0.01)
        else:
            lat = np.random.normal(base_lat, 0.1)
            lon = np.random.normal(base_lon, 0.1)
            
        # Random date and time within the last 2 years
        random_days = random.randint(0, 730)
        random_seconds = random.randint(0, 86400)
        crime_date = start_date + timedelta(days=random_days, seconds=random_seconds)
        
        # Correlate some crime types with weapons and times to ensure association rules can be found
        hour = crime_date.hour
        if hour >= 22 or hour <= 4:
            c_type = random.choices(['Robbery', 'Burglary', 'Assault', 'Murder'], weights=[0.4, 0.3, 0.2, 0.1])[0]
        elif 9 <= hour <= 18:
            c_type = random.choices(['Theft', 'Fraud', 'Cybercrime'], weights=[0.5, 0.3, 0.2])[0]
        else:
            c_type = random.choice(crime_types)
            
        if c_type in ['Murder', 'Robbery']:
            weapon = random.choices(['Firearm', 'Knife', 'Blunt Object'], weights=[0.4, 0.4, 0.2])[0]
        elif c_type in ['Assault']:
            weapon = random.choices(['None', 'Blunt Object', 'Knife'], weights=[0.5, 0.3, 0.2])[0]
        elif c_type in ['Theft', 'Fraud', 'Cybercrime', 'Vandalism']:
            weapon = 'None'
        else:
            weapon = random.choice(weapons)
            
        records.append({
            'Date': crime_date.strftime('%Y-%m-%d'),
            'Time': crime_date.strftime('%H:%M:%S'),
            'Latitude': lat,
            'Longitude': lon,
            'Crime_Type': c_type,
            'Weapon_Used': weapon,
            'Arrest_Made': random.choice([True, False, False, False]) # 25% arrest rate
        })
        
    df = pd.DataFrame(records)
    df.to_csv('delhi_crime_data.csv', index=False)
    print(f"Generated {num_records} records and saved to 'delhi_crime_data.csv'")

if __name__ == '__main__':
    generate_crime_data()
