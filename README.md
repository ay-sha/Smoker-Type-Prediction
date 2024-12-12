# Smoking and Drinking Dataset Analysis

## Overview  
This project analyzes a smoking and drinking dataset to predict the smoking status (`SMK_stat_type_cd`) of individuals based on various features. The analysis utilizes Apache Spark on Google Colab, employing machine learning models like Logistic Regression, Decision Tree, and Random Forest to classify the dataset, with a focus on preprocessing, feature engineering, and model evaluation.

## Dataset  
The dataset contains information on individuals, including attributes such as age, height, weight, sex, and their smoking status (`SMK_stat_type_cd`). The goal is to predict the smoking status using these features.

## Features  
- **Data Cleaning**: Missing values were handled by casting numerical columns and using null checks.
- **Data Transformation**: Categorical features, like `sex`, were indexed using StringIndexer. Numerical features were scaled using MinMaxScaler.
- **Machine Learning Models**: Logistic Regression, Decision Tree, and Random Forest were used to train and predict the `SMK_stat_type_cd`.
- **Performance Evaluation**: Accuracy, precision, recall, and F1-score were computed to evaluate model performance. Additionally, confusion matrices were plotted for model assessment.

## Technologies Used  
- **Apache Spark**: For distributed data processing and handling large datasets.  
- **Google Colab**: Used to run the Spark environment and perform computations.  
- **Python Libraries**:  
  - Data Processing: PySpark  
  - Machine Learning: PySpark MLlib  
  - Data Visualization: Seaborn, Matplotlib  
  - Misc: `findspark` for Spark initialization

## Steps Performed  
1. **Data Loading and Preprocessing**:  
   - Loaded the dataset from Google Drive.  
   - Casted columns such as age, height, and weight to numerical values.  
   - Handled missing values and null entries.  
2. **Feature Engineering**:  
   - Used `StringIndexer` to convert categorical features into numeric values.  
   - Scaled features using MinMaxScaler to standardize the data.  
3. **Model Training**:  
   - Trained Logistic Regression, Decision Tree, and Random Forest models.  
   - Split data into training and testing datasets (80% train, 20% test).  
4. **Model Evaluation**:  
   - Evaluated models using accuracy, and plotted confusion matrices for a deeper understanding of model performance.  
   - Achieved perfect accuracy (1.0) for Logistic Regression, Decision Tree, and Random Forest on test data.

## Results  
- **Logistic Regression**:  
  - Accuracy: **1.0**  
  - Confusion Matrix: Shows perfect classification.  
- **Decision Tree Classifier**:  
  - Accuracy: **1.0**  
  - Confusion Matrix: Indicates high performance with no misclassifications.  
- **Random Forest Classifier**:  
  - Accuracy: **1.0**  
  - Confusion Matrix: Perfect classification with all predictions correct.

## Visualizations  
- **Confusion Matrix**:  
  A heatmap to visualize the confusion matrix for each model, showcasing the accuracy and model performance.  

  Example Confusion Matrix for Random Forest:  
![Capture](https://github.com/user-attachments/assets/22c2effc-b160-4971-84ee-24a0cf78d1a3)


## How to Use  
1. Clone this repository or run in a Google Colab notebook.  
2. Ensure you have access to the dataset and load it using the provided code.  
3. Install necessary libraries:  
   ```bash  
   pip install pyspark findspark seaborn matplotlib  
   ```  
4. Run the notebook to perform preprocessing, training, and evaluation.
