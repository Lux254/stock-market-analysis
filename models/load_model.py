from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('models/lstm_stock_model.h5')

# Make predictions
predictions = model.predict(input_data)