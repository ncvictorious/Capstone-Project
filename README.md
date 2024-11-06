# Capstone Project: Customer Churn Prediction using Machine Learning

## Project Overview
This project applies machine learning techniques to predict customer churn in a telecommunications dataset. The goal is to build and evaluate classification models that help identify customers likely to churn, providing insights that can drive proactive retention strategies.

This project is a part of my MSc program and showcases my skills in data analysis, feature engineering, and model evaluation. I have implemented several classification algorithms, including Logistic Regression, Random Forest, and Decision Trees, and optimized them for best performance.

## Key Features
- **Comprehensive Exploratory Data Analysis (EDA)**: A detailed analysis of customer demographics and service usage patterns to understand factors affecting churn.
- **Modeling and Optimization**: Includes multiple classifiers and hyperparameter tuning to find the best model for churn prediction.
- **Feature Importance Analysis**: Insights into which features most influence the likelihood of churn, helping to target retention efforts effectively.

## Repository Structure
- **data/**: Contains the dataset used for this project (placeholder for user to add their own data if needed).
- **notebooks/**: Jupyter notebooks with the full analysis, including EDA, model training, and evaluation.
- **images/**: Visualizations created during the project, such as correlation matrices, feature importance charts, and confusion matrices.
- **scripts/**: Python scripts for model training, testing, and performance evaluation.
- **README.md**: Overview and instructions for the project.

## Dataset
The dataset consists of customer information such as demographics, account information, and service usage. Each row represents a customer, and each column contains attributes related to customer behavior and account status.

### Target Variable
- **Churn**: A binary variable indicating whether the customer left within the last month (1 for churned, 0 for retained).

## Installation Instructions
To run this project locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Capstone-Project.git
   cd Capstone-Project
Install required packages: Ensure Python and pip are installed, then run:

bash
Copy code
pip install -r requirements.txt
Run Jupyter Notebooks: Open the notebooks in the notebooks/ directory to view the full analysis and model implementation:

bash
Copy code
jupyter notebook
Project Workflow
Exploratory Data Analysis (EDA):

Visualized relationships between features such as contract type, tenure, monthly charges, and churn.
Generated correlation matrices and scatter plots to identify patterns in customer behavior.
Feature Engineering:

Scaled numerical features using MinMaxScaler for model optimization.
Selected top features that significantly impact churn prediction using feature importance techniques.
Model Training and Evaluation:

Logistic Regression:
Accuracy: 76.9%
Precision, Recall, and F1-Score provided in the notebook.
Random Forest:
Accuracy: 76.7%
Provides a balance of interpretability and accuracy with insights into feature importance.
Decision Tree:
Accuracy: 74.4%
Visualized decision paths for straightforward interpretability.
SGD Classifier:
Accuracy: 74.7%
Evaluated for performance with large datasets and high-dimensional data.
Each model’s performance was evaluated using accuracy, precision, recall, F1-score, and confusion matrices.

Model Optimization:

Fine-tuned hyperparameters for each model to improve performance.
Enhanced performance by focusing on precision and recall, particularly for the churn class (1), to minimize false negatives.
Key Visualizations
Confusion Matrices: Depict each model’s true positives, false positives, true negatives, and false negatives.
Feature Importance: Shows which features most strongly influence churn prediction, including contract type, tenure, and monthly charges.
Correlation Matrix: Identifies relationships between features to avoid multicollinearity.
Results Summary
The project demonstrates that using machine learning for churn prediction can provide telecom companies with actionable insights for customer retention. The Random Forest model performed best in balancing interpretability and predictive power. Key findings include:

Contract type and tenure are strong indicators of churn.
Monthly charges and internet service type also significantly influence customer retention.
Future Work
Implement Deep Learning Models: Further improve accuracy with neural networks.
Deploy as a Web Application: Use frameworks like Flask or Django to deploy the model for real-time predictions.
Incorporate Additional Data: Enhance the model by integrating external data sources, such as customer support interactions or marketing data.
How to Contribute
If you'd like to contribute to this project, feel free to submit a pull request or report an issue.

Contact Information
For more details, contact me via nwaobi.victor@gmail.com or connect with me on LinkedIn:https://www.linkedin.com/in/victor-nwaobi-738583174/.
