
# Wine Quality Classification using UCI Wine Dataset

## Project Overview
This project aims to classify wine as either "Good" or "Not Good" based on its chemical properties using a neural network model. The dataset used is the Wine Quality Dataset from the UCI Machine Learning Repository, which includes information on various chemical properties of red wine and their corresponding quality scores. By predicting wine quality based on these properties, this project seeks to assist wine producers in quality control and improve consumer satisfaction.

## Project Objectives
1. Analyze the chemical composition of wines to identify correlations with wine quality.
2. Classify wines into two categories:
   - **Good Quality Wine**: Quality score ≥ 7
   - **Not Good Quality Wine**: Quality score < 7
3. Develop a neural network model to predict wine quality based on the chemical properties.

## Dataset
The dataset is sourced from the UCI Machine Learning Repository and includes 1,599 samples of red wine. Each sample includes 11 chemical properties and a quality score (0-10). Key features include:
- **fixed acidity**
- **volatile acidity**
- **citric acid**
- **residual sugar**
- **chlorides**
- **free sulfur dioxide**
- **total sulfur dioxide**
- **density**
- **pH**
- **sulphates**
- **alcohol**

For classification, the quality score is binarized into "Good" (score ≥ 7) and "Not Good" (score < 7).

## Project Workflow
The project follows these main steps:
1. **Data Collection**: Load the dataset from the UCI repository.
2. **Data Preprocessing**: 
   - Binarize the target variable to create a binary classification problem.
   - Scale the features using `StandardScaler` to improve model performance.
3. **Exploratory Data Analysis (EDA)**: 
   - Generate summary statistics and visualize feature distributions.
   - Analyze correlations between features and the quality label.
4. **Model Building**: Develop a feedforward neural network with two hidden layers using Keras. Dropout layers can be added to prevent overfitting.
5. **Model Evaluation**: Evaluate the model using metrics such as accuracy, precision, recall, F1-score, and confusion matrix.
6. **Data Visualization**: Visualize distribution of quality labels, feature correlations, and model performance over epochs.

## Model Architecture
The neural network model includes:
- **Input Layer**: 64 neurons with ReLU activation.
- **Hidden Layer 1**: 64 neurons with ReLU activation.
- **Hidden Layer 2**: 32 neurons with ReLU activation.
- **Output Layer**: 1 neuron with sigmoid activation for binary classification.

The model is compiled using the `adam` optimizer and `binary_crossentropy` loss, and trained for 50 epochs with a batch size of 16.

## Results
The model achieves a reasonable balance between accuracy and generalization:
- **Training Accuracy**: ~95%
- **Test Accuracy**: ~87%
- **Evaluation Metrics**: 
   - Precision
   - Recall
   - F1 Score
   - Confusion Matrix

Due to the dataset's class imbalance (fewer "Good" wines), the F1 Score and confusion matrix provide valuable insight into the model's performance.

## Installation and Setup
To run this project locally, follow these steps:
1. Clone this repository:
   git clone https://github.com/likhiths23/Wine-Quality-Classification.git
   
2. Install the required packages:
   pip install -r requirements.txt

3. Run the Jupyter Notebook to train and evaluate the model.

## Dependencies
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow/Keras

## Files in Repository
- `Wine_Quality_Classification.ipynb`: Jupyter Notebook with all code and analysis.
- `README.md`: Overview of the project and instructions.
- `requirements.txt`: List of required Python packages.

## Future Work
Further improvements to the model and analysis could include:
- Using additional feature engineering techniques.
- Experimenting with other neural network architectures or machine learning models.
- Applying techniques to address the class imbalance, such as SMOTE or class weighting.

## Conclusion
This project demonstrates a method for automating wine quality classification using neural networks. The analysis and model provide insights into the chemical factors that influence wine quality and offer a potential solution for quality assessment in winemaking.
