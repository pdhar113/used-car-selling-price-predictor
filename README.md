# Used Car Selling Price Predictor

A machine learning-powered Flask web application that predicts used car selling prices based on vehicle characteristics.

## Features

- **Interactive Web Interface**: User-friendly form to input car details
- **ML Model**: Trained model using scikit-learn for accurate price predictions
- **Model Confidence**: Displays R² score indicating prediction reliability
- **Real-time Predictions**: Get instant price estimates

## Project Structure

```
.
├── app.py                          # Flask application
├── used_car_selling_price_model.pkl # Trained ML model
├── ml/                             # Machine learning pipeline scripts
│   ├── Step1-load-training-dataset.py
│   ├── Step2-features-target.py
│   ├── Step3-train-model.py
│   ├── Step4-predic-terminal-after-model-created.py
│   ├── Step5-evaluate-with-test-dataset.py
│   ├── Step6-save-model.py
│   ├── Step7-predict-with-saved-model.py
│   ├── encoder.py                  # Categorical data encoder
│   └── splitter.py                 # Train-test data splitter
├── templates/                      # HTML templates
│   ├── index.html                  # Prediction form
│   └── result.html                 # Prediction result page
├── orig-dataset.csv                # Original dataset
├── encoded-dataset.csv             # Encoded dataset
├── train.csv                       # Training data
├── test.csv                        # Test data
└── requirements.txt                # Python dependencies
```

## Installation

### Prerequisites
- Python 3.11+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/pdhar113/used-car-selling-price-predictor.git
   cd used-car-selling-price-predictor
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the Flask Application

```bash
python app.py
```

The application will start at `http://localhost:5000`

### Make a Prediction

1. Open your browser and navigate to `http://localhost:5000`
2. Fill in the car details:
   - **Car Name**: Select from the available models
   - **Vehicle Age**: Years since purchase (0-29)
   - **Kilometers Driven**: Total distance traveled (0-4,000,000)
   - **Seller Type**: Individual, Dealer, or Trustmark Dealer
   - **Fuel Type**: Petrol, Diesel, CNG, LPG, or Electric
   - **Transmission**: Manual or Automatic
3. Click "Predict Price" to get the estimated selling price
4. View the predicted price and model confidence score

## Machine Learning Pipeline

The project includes a complete ML pipeline:

1. **Data Loading** (`Step1`): Load the original dataset
2. **Feature Engineering** (`Step2`): Extract features and target variable
3. **Model Training** (`Step3`): Train the regression model
4. **Terminal Prediction** (`Step4`): Quick predictions from command line
5. **Model Evaluation** (`Step5`): Evaluate model on test data
6. **Model Serialization** (`Step6`): Save trained model to disk
7. **Inference** (`Step7`): Load and use saved model for predictions

### Data Processing

- **Encoding** (`encoder.py`): Converts categorical variables to numerical codes
- **Splitting** (`splitter.py`): Splits data into 70% training and 30% testing sets

## Model Performance

The model uses R² scoring to indicate prediction reliability. The confidence score is displayed with each prediction.

- **Training Data**: 70% of the dataset
- **Test Data**: 30% of the dataset
- **Random State**: 42 (for reproducibility)

## Input Validation

The application validates all inputs:
- Vehicle age: 0-29 years
- Kilometers driven: 0-4,000,000 km
- All categorical fields must be selected from provided options

## Dependencies

See `requirements.txt` for the complete list. Key dependencies:

- **Flask**: Web framework for the application
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning models and evaluation
- **joblib**: Model serialization
- **numpy**: Numerical computing (≥1.26.4)
- **scipy**: Scientific computing

## Troubleshooting

### ModuleNotFoundError: numpy.exceptions
This indicates a NumPy version conflict. Ensure NumPy ≥1.26.4 is installed:
```bash
pip install --upgrade "numpy>=1.26.4"
```

### Model not found
Ensure `used_car_selling_price_model.pkl` exists in the project root directory.

### Port already in use
If port 5000 is already in use, you can specify a different port:
```bash
python -c "from app import app; app.run(port=5001, debug=True)"
```

## Environment Variables

- `FLASK_DEBUG`: Set to "1" to enable debug mode (default: disabled)

Example:
```bash
set FLASK_DEBUG=1
python app.py
```

## License

This project is open source and available under the MIT License.

## Author

Prithviraj Dhar (@pdhar113)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/pdhar113/used-car-selling-price-predictor/issues).

## Project Links

- **GitHub Repository**: [https://github.com/pdhar113/used-car-selling-price-predictor](https://github.com/pdhar113/used-car-selling-price-predictor)
- **Flask Documentation**: [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)
- **scikit-learn Documentation**: [https://scikit-learn.org/](https://scikit-learn.org/)
