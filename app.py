import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Crime Pattern Detection", layout="wide")

st.title("🚓 Crime Pattern Detection - Interactive Dashboard")
st.markdown("This dashboard presents insights from the Delhi Crime Pattern Detection dataset, featuring spatial clustering and association rule mining.")

# Load Data
@st.cache_data
def load_data():
    if os.path.exists('delhi_crime_data.csv'):
        return pd.read_csv('delhi_crime_data.csv')
    return pd.DataFrame()

@st.cache_data
def load_rules():
    if os.path.exists('association_rules_report.csv'):
        return pd.read_csv('association_rules_report.csv')
    return pd.DataFrame()

df = load_data()
rules = load_rules()

if df.empty:
    st.error("Data not found! Please run the dataset generator first.")
else:
    # Sidebar Filters
    st.sidebar.header("Filter Data")
    crime_types = ['All'] + list(df['Crime_Type'].unique())
    selected_crime = st.sidebar.selectbox("Select Crime Type", crime_types)
    
    if selected_crime != 'All':
        df = df[df['Crime_Type'] == selected_crime]
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Crime Distribution ({selected_crime})")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, y='Crime_Type', palette='viridis', ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.subheader("Time of Day Analysis")
        # Creating Time_of_Day on the fly just in case
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df['Hour'] = df['Datetime'].dt.hour
        def cat_time(h):
            if 5 <= h <= 11: return 'Morning'
            elif 12 <= h <= 16: return 'Afternoon'
            elif 17 <= h <= 20: return 'Evening'
            else: return 'Night'
        df['Time_of_Day'] = df['Hour'].apply(cat_time)
        
        fig2, ax2 = plt.subplots()
        sns.countplot(data=df, x='Time_of_Day', order=['Morning', 'Afternoon', 'Evening', 'Night'], palette='magma', ax=ax2)
        st.pyplot(fig2)

    st.subheader("Geographical Crime Hotspots")
    st.markdown("Using K-Means clustering, the following clusters were generated. The scatter plot maps the geographic spread (Longitude vs Latitude).")
    # Quick clustering for the filtered data
    if len(df) > 5:
        from scipy.cluster.vq import kmeans2
        coords = df[['Latitude', 'Longitude']].values
        centers, labels = kmeans2(coords, min(5, len(df)), minit='points')
        df['Cluster'] = labels
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='Longitude', y='Latitude', hue='Cluster', palette='tab10', ax=ax3)
        st.pyplot(fig3)
    else:
        st.write("Not enough data to cluster.")

    if not rules.empty and selected_crime == 'All':
        st.subheader("Top Correlated Crime Patterns (Association Rules)")
        st.markdown("These rules were discovered using the Apriori algorithm.")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
