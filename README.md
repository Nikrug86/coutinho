# Heart Disease Prediction Project

This project demonstrates a machine learning pipeline for predicting heart disease using various classification models, including Logistic Regression, Random Forest, and a Neural Network. The project includes data preprocessing, model training, and an inference script for making predictions on new data.

## Project Structure

- `train.csv`: Training dataset.
- `test.csv`: Test dataset for predictions.
- `model_artifacts/`: Directory to store trained model weights and preprocessing tools.
  - `nn_model_weights.weights.h5`: Weights for the trained Neural Network model.
  - `scaler.pkl`: Pre-fitted `StandardScaler` object used for numerical feature scaling.
- `inference.py`: A Python script for loading the trained Neural Network model and scaler, preprocessing new data, and making predictions.
- `requirements.txt`: Lists all Python dependencies required to run the project.
- `README.md`: This file.

## Setup Instructions

To set up and run this project locally, follow these steps:

### 1. Clone the Repository

```bash
git clone <repository_url>
cd heart-disease-prediction
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install all required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Data

Ensure `train.csv` and `test.csv` are in the project root directory. These files are typically provided in competition environments or as part of the dataset.

## Running the Inference Script

The `inference.py` script allows you to make predictions on new data using the trained Neural Network model.

### 1. Ensure Model Artifacts are Present

Before running `inference.py`, make sure the `model_artifacts/` directory exists and contains `nn_model_weights.weights.h5` and `scaler.pkl`. These files are generated after training the Neural Network model (if you ran the training notebook).

### 2. Prepare Your Input Data

The `inference.py` script expects new data in a pandas DataFrame format, similar to the original `test.csv` (excluding the `ID` and `class` columns). An example of how to structure this data is provided within the `inference.py` script itself.

### 3. Run the Script

Execute the `inference.py` script from your terminal:

```bash
python inference.py
```

The script will load the model and scaler, preprocess the example data provided within the script, and print the predicted labels and probabilities.

### Example Usage within Python

You can also import and use the `predict` function from `inference.py` in your own Python scripts:

```python
import pandas as pd
from inference import predict

# Example new data (replace with your actual data)
new_data = pd.DataFrame([
    {
        'age': 63,
        'sex': 1,
        'chest': 3,
        'resting_blood_pressure': 145,
        'serum_cholestoral': 233,
        'fasting_blood_sugar': 1,
        'resting_electrocardiographic_results': 0,
        'maximum_heart_rate_achieved': 150,
        'exercise_induced_angina': 0,
        'oldpeak': 2.3,
        'slope': 0,
        'number_of_major_vessels': 0,
        'thal': 1
    }
])

predictions, probabilities = predict(new_data)

print("Predictions:", predictions)
print("Probabilities:", probabilities)
```

## Contact

For any questions or suggestions, please open an issue in the GitHub repository.
"""
