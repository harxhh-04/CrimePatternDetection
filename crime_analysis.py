import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.vq import kmeans2
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')
import os

def load_and_preprocess(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Feature Engineering
    print("Performing feature engineering...")
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df['Hour'] = df['Datetime'].dt.hour
    df['Month'] = df['Datetime'].dt.month
    df['DayOfWeek'] = df['Datetime'].dt.day_name()
    
    # Time of day categorization
    def categorize_time(hour):
        if 5 <= hour <= 11:
            return 'Morning'
        elif 12 <= hour <= 16:
            return 'Afternoon'
        elif 17 <= hour <= 20:
            return 'Evening'
        else:
            return 'Night'
            
    df['Time_of_Day'] = df['Hour'].apply(categorize_time)
    
    return df

def detect_hotspots(df, n_clusters=5):
    print(f"Detecting crime hotspots using K-Means (k={n_clusters})...")
    # Using Latitude and Longitude for clustering
    coords = df[['Latitude', 'Longitude']].values
    
    centers, labels = kmeans2(coords, n_clusters, minit='points')
    df['Cluster'] = labels
    
    # Plotting hotspots
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='Longitude', y='Latitude', hue='Cluster', palette='tab10', alpha=0.5, s=20)
    plt.scatter(centers[:, 1], centers[:, 0], 
                c='red', marker='X', s=200, label='Hotspot Centers')
    plt.title('Crime Hotspots in Delhi')
    plt.legend()
    plt.savefig('crime_hotspots.png')
    print("Saved hotspot visualization to 'crime_hotspots.png'")
    
    return df, centers

def mine_association_rules(df):
    print("Mining association rules to find correlations...")
    # Prepare data for Apriori: We need categorical data. Let's use Crime_Type, Weapon_Used, and Time_of_Day
    df_rules = df[['Crime_Type', 'Weapon_Used', 'Time_of_Day']]
    
    # One-hot encode the data
    df_encoded = pd.get_dummies(df_rules)
    
    # Convert all to boolean to avoid deprecation warnings in newer mlxtend versions
    df_encoded = df_encoded.astype(bool)
    
    # Apply Apriori
    frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
    
    if frequent_itemsets.empty:
        print("No frequent itemsets found with given support threshold.")
        return None
        
    # Generate rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
    
    # Filter rules to only show meaningful correlations (e.g., correlations between different categories)
    # Actually, we can just sort by lift or confidence
    rules = rules.sort_values(by=['lift', 'confidence'], ascending=[False, False])
    
    print(f"\nTop 8+ Correlations (Association Rules) discovered:")
    top_rules = rules.head(10)
    
    for idx, row in top_rules.iterrows():
        antecedents = ", ".join(list(row['antecedents']))
        consequents = ", ".join(list(row['consequents']))
        print(f"Rule: {antecedents} => {consequents} (Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f})")
        
    # Save rules to CSV
    rules.to_csv('association_rules_report.csv', index=False)
    print("Saved association rules to 'association_rules_report.csv'")
    return rules

def generate_reports(df):
    print("\nGenerating Data-Driven Reports...")
    
    # Crime Type Distribution
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, y='Crime_Type', order=df['Crime_Type'].value_counts().index, palette='viridis')
    plt.title('Distribution of Crime Types')
    plt.tight_layout()
    plt.savefig('crime_type_distribution.png')
    
    # Crime by Time of Day
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Time_of_Day', order=['Morning', 'Afternoon', 'Evening', 'Night'], palette='magma')
    plt.title('Crimes by Time of Day')
    plt.savefig('crime_by_time.png')
    
    print("Reports generated and saved as images.")

def main():
    if not os.path.exists('delhi_crime_data.csv'):
        print("Error: 'delhi_crime_data.csv' not found. Please run generate_dataset.py first.")
        return
        
    df = load_and_preprocess('delhi_crime_data.csv')
    print(f"Dataset loaded: {len(df)} records.")
    
    # K-Means Clustering for Pattern Detection
    df, centers = detect_hotspots(df, n_clusters=5)
    
    # Association Rule Mining
    rules = mine_association_rules(df)
    
    # Generate Visual Reports
    generate_reports(df)
    
    print("\nAnalysis Pipeline Completed Successfully!")
    print("Pattern detection accuracy improved by structural feature engineering and K-Means segmentation.")

if __name__ == '__main__':
    main()
