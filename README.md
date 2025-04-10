
# Credit Card Fraud Detection Streamlit App

## Project Overview

This is a Streamlit-based web application for detecting fraudulent credit card transactions using a machine learning model. It allows users to upload a CSV file containing transaction data, and the app predicts whether a transaction is fraudulent or legitimate. It also displays interactive charts like the confusion matrix, ROC curve, and fraud probability distribution.

## Features

- **CSV Upload**: Users can upload a CSV file with transaction data.
- **Fraud Prediction**: The model predicts whether each transaction is fraudulent or legitimate.
- **Visualizations**:
    - Confusion Matrix
    - ROC Curve
    - Fraud Probability Distribution
- **Evaluation Metrics**: View precision, recall, F1 score, and accuracy (if 'Class' column is present in the uploaded data).

## Dataset

The application uses a dataset of credit card transactions. You can use the provided CSV or upload your own file with the same format. The dataset should include columns `Time`, `Amount`, and optionally `Class`.

**Example columns**:
- `Time`: The time elapsed since the first transaction.
- `Amount`: The transaction amount.
- `Class`: Fraud label (1 for fraud, 0 for legitimate).

## Model Details

The model used in this app is a RandomForestClassifier trained on historical transaction data. It predicts the likelihood of fraud in each transaction.

## Installation

To run the app locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd credit-card-fraud-app
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

4. Visit the app in your browser at `http://localhost:8501`.

## Usage

- Upload a CSV file with transaction data. 
- The app will predict whether each transaction is fraudulent or legitimate and display the results.
- View confusion matrix, ROC curve, and fraud probability distribution.
- Optionally, evaluate the model using the ground truth from the 'Class' column.

## Example File Format

| Time    | Amount | Class  |
|---------|--------|--------|
| 100.1   | 12.45  | 0      |
| 150.2   | 10.00  | 1      |
| 130.5   | 50.75  | 0      |
| ...     | ...    | ...    |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Dataset**: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Libraries**: Streamlit, scikit-learn, pandas, matplotlib, seaborn, joblib, imbalanced-learn
