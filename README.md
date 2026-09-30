# CrimePatternDetection

An end-to-end data analysis pipeline for crime records, focusing on the Delhi (India) region. 
This project leverages K-Means clustering and association rule mining to identify crime hotspots and correlations between crime types, weapons, and time of day.

## Features
- **Dataset Generation**: Synthetic generation of 12,000+ crime records mirroring real-world constraints for Delhi.
- **Feature Engineering**: Temporal extraction (hour, time of day) for deeper pattern recognition.
- **Hotspot Detection**: Uses K-Means clustering (k=5) to pinpoint 5 key geographical crime hotspots.
- **Association Rule Mining**: Uses the Apriori algorithm to discover correlations (e.g., specific crimes at night with specific weapons).
- **Data Visualization**: Generated reports and plots using Matplotlib and Seaborn for data-driven law enforcement decisions.

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate the dataset:
   ```bash
   python generate_dataset.py
   ```

3. Run the analysis pipeline:
   ```bash
   python crime_analysis.py
   ```

## Results
- `delhi_crime_data.csv`: The generated 12,000+ record dataset.
- `crime_hotspots.png`: Map of crime clusters (hotspots).
- `crime_type_distribution.png`: Bar chart of overall crime types.
- `crime_by_time.png`: Crime frequency grouped by time of day.
- `association_rules_report.csv`: Discovered rules and correlations between dimensions.
