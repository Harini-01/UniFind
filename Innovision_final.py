import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load the college dataset (replace 'your_colleges_data.csv' with your actual file)
file_path = r"output_file.xls"
df = pd.read_csv(file_path)

# Function to preprocess data and compute similarity matrix
def preprocess_and_compute_similarity(df):
    # Select relevant columns for recommendation (e.g., 'Fees' and 'City')
    selected_columns = ['Average Fees', 'City']

    # Drop rows with missing values in selected columns
    df_selected = df[selected_columns].dropna()

    # Scale numerical columns to ensure fair comparison
    scaler = MinMaxScaler()
    df_selected[' Average Fees'] = scaler.fit_transform(df_selected[['Average Fees']])

    # Convert categorical columns to one-hot encoding
    df_encoded = pd.get_dummies(df_selected, columns=['City'])

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(df_encoded)
    
    return similarity_matrix

# Streamlit app
def main():
    st.title("UniFind")

    # User input
    fees_preference = st.slider('Select your preferred fees range:', min_value=0, max_value=2000000, step=1000, value=(0, 2000000))
    city_preference = st.selectbox('Select your preferred city:', df['City'].unique())

    # Preprocess data and compute similarity matrix
    similarity_matrix = preprocess_and_compute_similarity(df)

    # Recommend colleges based on user preferences
    recommendations = get_recommendations(fees_preference, city_preference, df, similarity_matrix)
    
    # Display recommendations
    st.subheader('Recommended Colleges:')
    st.table(recommendations)

# Function to get recommendations
def get_recommendations(fees_preference, city_preference, df, similarity_matrix):
    # Filter colleges based on user preferences
    filtered_df = df[(df['Average Fees'] >= fees_preference[0]) & (df['Average Fees'] <= fees_preference[1]) & (df['City'] == city_preference)]

    if filtered_df.empty:
        return pd.DataFrame(columns=df.columns)  # Return an empty DataFrame if no matches found

    # Compute similarity scores for filtered colleges
    similarity_scores = similarity_matrix[filtered_df.index].sum(axis=0)

    # Sort colleges based on similarity scores
    sorted_indices = similarity_scores.argsort()[::-1]
    sorted_colleges = df.iloc[sorted_indices]

    sorted_colleges = sorted_colleges.sort_values(by='Rating', ascending=False)
    return sorted_colleges

if __name__ == '__main__':
    main()
