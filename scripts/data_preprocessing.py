import pandas as pd
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt

def convert_to_datetime(data, date_column):
    """
    Converts a specified column to datetime format.
    """
    data[date_column] = pd.to_datetime(data[date_column])
    return data

def remove_outliers(data, threshold=3):
    """
    Removes rows with outliers based on Z-score.
    """
    z_scores = data.apply(zscore)
    outliers = (z_scores.abs() > threshold).any(axis=1)
    return data[~outliers]

def plot_correlation_heatmap(data):
    """
    Plots a heatmap of the correlation matrix.
    """
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        linewidths=0.5
    )
    plt.title('Correlation Heatmap')
    plt.show()

if __name__ == "__main__":
    # Example usage
    yahoo_data = pd.read_excel(r"C:\Users\USER\Downloads\yahoo_data.xlsx")  # Load your raw data
    yahoo_data = convert_to_datetime(yahoo_data, "Date")
    
    # Save the Date column separately if needed
    date_column = yahoo_data['Date']
    features = yahoo_data.drop(columns=['Date'])
    
    # Remove outliers
    clean_data = remove_outliers(features)
    
    # Optional: Plot correlation heatmap
    plot_correlation_heatmap(clean_data)
    
    # Save preprocessed data
    clean_data.to_csv(r"C:\Users\USER\OneDrive\Dokumenter\data\processed\clean_stock_data.xlsx", index=False)
    print("Data preprocessing complete.")
