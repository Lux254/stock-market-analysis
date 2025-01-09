# Stock Market Price Prediction Using LSTM with Technical Indicators

This project involves developing a predictive model for forecasting stock prices using historical data and technical indicators. The Long Short-Term Memory (LSTM) neural network was employed for its ability to capture temporal dependencies, making it ideal for time-series forecasting tasks like stock price prediction.

---

## Features
- Time-series forecasting using LSTM.
- Integration of technical indicators, including Exponential Moving Average (EMA).
- Comprehensive data preprocessing, including outlier removal and feature scaling.
- Evaluation using standard regression metrics: MAE, RMSE, and R².

---

## Data Source
The dataset was sourced from Yahoo Finance, containing:
- Historical stock prices: Open, High, Low, Close, Adjusted Close.
- Trading volume.

---

## Installation
To replicate this project:

1. Clone the repository:
   ```bash
   git clone https://github.com/username/stock-market-prediction.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the training script:
   ```bash
   python train_model.py
   ```

---

## Usage
- Preprocessed historical stock data is used as input.
- The model predicts the `Close` price based on past trends.
- Results are visualized for analysis.

---

## Results
The model achieved:
- **Mean Absolute Error (MAE):** 234.24
- **Root Mean Squared Error (RMSE):** 317.13
- **R-squared (R²):** 0.86

Incorporating the Exponential Moving Average (EMA) improved prediction accuracy significantly, with a 50% reduction in error metrics compared to models without feature engineering.

---

## Visualizations
![Actual vs Predicted Stock Prices] ![alt text](23.11.2024_04.21.38_REC.png)

This plot demonstrates the model's ability to closely follow the trends in stock price movements.

---

## Model Architecture
The LSTM-based model includes:
- Two LSTM layers with 50 units each.
- Dropout layers (rate: 0.2) to prevent overfitting.
- A Dense output layer for predicting the `Close` price.

Optimizer: Adam | Loss Function: Mean Squared Error (MSE)

---

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## Contact
For questions or collaborations:
- **Email:** shadrackohungo@gmail.com
- **LinkedIn:** [https://www.linkedin.com/in/shadrack-omondi-30071a1a4/]

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.
