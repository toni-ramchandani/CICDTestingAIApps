import os
import re
import pickle
import numpy as np
import pandas as pd
import warnings
import logging

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from giskard import Dataset, Model, scan, testing

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

CATEGORICAL_COLUMNS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
NUMERIC_COLUMNS = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Load the validation dataset
logging.info("Loading the validation dataset")
validation_df = pd.read_csv('data/validation_data.csv')
validation_df[CATEGORICAL_COLUMNS] = validation_df[CATEGORICAL_COLUMNS].astype('object')
FEATURES = [col for col in validation_df.columns if col != 'Churn']

# Load the model
logging.info("Loading the model")
with open("model/model.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Wrap the dataset with Giskard
logging.info("Wrapping the dataset with Giskard")
wrapped_data = Dataset(
    df = validation_df,  # A pandas.DataFrame that contains the raw data (before all the pre-processing steps) and the actual ground truth variable
    target = 'Churn',  # Ground truth variable
    name = "Churn classification dataset",  # Optional
    cat_columns = CATEGORICAL_COLUMNS  # List of categorical columns. Optional, but is a MUST if available. Inferred automatically if not.
)

# Define the prediction function
logging.info("Defining the prediction function")
def prediction_function(df: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(df)

# Wrap the model with Giskard
logging.info("Wrapping the model with Giskard")
wrapped_model = Model(
    model = prediction_function,                # A prediction function that encapsulates all the data pre-processing steps and that could be executed with the dataset used by the scan.
    model_type = "classification",              # Either regression, classification or text_generation.
    classification_labels = pipeline.classes_,  # Their order MUST be identical to the prediction_function's output order
    name = "Churn classification",              # Name of the wrapped model [Optional]
    feature_names = FEATURES,                   # Default: all columns of your dataset [Optional]
    classification_threshold = 0.5              # Default: 0.5 [Optional]
)

# Scan the model
logging.info("Scanning the model")
scan_results = scan(wrapped_model, wrapped_data)

# Create a test suite from the scan results and add custom tests
logging.info("Creating a test suite from the scan results and adding custom tests")
test_suite = scan_results.generate_test_suite("My first test suite")
test_suite = test_suite.add_test(testing.test_accuracy(wrapped_model, wrapped_data, threshold=0.75))
test_suite_results = test_suite.run()

if scan_results.has_issues():
    print("Your model has vulnerabilities")
else:
    print("Your model is safe")

# Extract the values of the test suite results using the `results` attribute
logging.info("Extracting the values of the test suite results using the `results` attribute")
output = dict()
for idx, test_result in enumerate(test_suite_results.results):
    test_name = re.sub('"|`|"|"', "", test_result[0])
    output[test_name] = {
        "Status": test_result[1].passed,
        "Threshold": test_result[2]["threshold"],
        "Score": test_result[1].metric,
    }

# To log the results to a pull request comment,
# save the results as a GitHub environment variable
logging.info("Saving the results as a GitHub environment variable")
import json
with open(os.getenv("GITHUB_ENV"), 'a') as fh:
    fh.write(f'TEST_RESULT={json.dumps(output)}')
