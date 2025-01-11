import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_atr(data, high_col, low_col, close_col, atr_period=14):
    """
    Calculates the Average True Range (ATR) for the dataset.
    """
    data = data.copy()
    data['High-Low'] = data[high_col] - data[low_col]
    data['High-Close'] = abs(data[high_col] - data[close_col].shift(1))
    data['Low-Close'] = abs(data[low_col] - data[close_col].shift(1))
    data['TR'] = data[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    data[f'ATR_{atr_period}'] = data['TR'].rolling(window=atr_period).mean()
    data.drop(columns=['High-Low', 'High-Close', 'Low-Close', 'TR'], inplace=True)
    return data

def calculate_ema(data, column, ema_period=10):
    """
    Calculates the Exponential Moving Average (EMA) for the dataset.
    """
    data = data.copy()
    data[f'EMA_{ema_period}'] = data[column].ewm(span=ema_period, adjust=False).mean()
    return data

def calculate_rsi(data, column, rsi_period=14):
    """
    Calculates the Relative Strength Index (RSI) for the dataset.
    """
    data = data.copy()
    delta = data[column].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    data[f'RSI_{rsi_period}'] = 100 - (100 / (1 + rs))
    return data

def plot_correlation_heatmap(data, exclude_columns=None):
    """
    Plots a heatmap for the correlation matrix.
    """
    if exclude_columns:
        data = data.drop(columns=exclude_columns)
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
    clean_data = pd.read_excel(r"C:\Users\USER\OneDrive\Dokumenter\data\processed\clean_stock_data.xlsx")
    
    # Calculate features
    clean_data = calculate_atr(clean_data, high_col="High", low_col="Low", close_col="Close*", atr_period=14)
    clean_data = calculate_ema(clean_data, column="Adj Close**", ema_period=10)
    clean_data = calculate_rsi(clean_data, column="Adj Close**", rsi_period=14)
    
    # Optional: Plot correlation heatmap
    plot_correlation_heatmap(clean_data, exclude_columns=['Date', 'Open', 'High', 'Low', 'Adj Close**'])
    
    # Save the dataset with new features
    clean_data.to_csv(r"C:\Users\USER\OneDrive\Dokumenter\data\processed/feature_engineered_data.csv", index=False)
    print("Feature engineering complete.")
