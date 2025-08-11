# E-Nose Cocoa Bean Classification
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/ahmadfauzana/enose)

This repository contains a comprehensive Python script for classifying cocoa bean samples using data from an electronic nose (e-nose). The project employs various machine learning algorithms to classify samples into predefined categories and predicts the classification for new, unclassified samples.

The script performs a complete machine learning workflow, including data loading, exploratory data analysis (EDA), model training, hyperparameter tuning, prediction on unseen data, and detailed visualization of the results.

## Features
- **Data Preprocessing:** Loads and cleans e-nose sensor data from an Excel file.
- **Exploratory Data Analysis (EDA):** Generates a suite of visualizations to understand the dataset, including:
    - Class distribution plots
    - Feature correlation heatmaps
    - Sensor reading distributions (boxplots)
    - PCA for dimensionality reduction and visualization
    - Feature importance analysis using a Random Forest
- **Multi-Model Training:** Trains and evaluates eight different machine learning classifiers:
    - Random Forest
    - Support Vector Machine (SVM)
    - Logistic Regression
    - Gradient Boosting
    - K-Nearest Neighbors (KNN)
    - Gaussian Naive Bayes
    - Multi-layer Perceptron (MLP)
    - Decision Tree
- **Hyperparameter Tuning:** Automatically tunes the best-performing models (Random Forest, SVM, Gradient Boosting) using `GridSearchCV`.
- **Prediction on Unseen Data:** Classifies 10 unclassified cocoa bean samples using all trained models (both original and tuned).
- **Results Visualization:** Creates detailed plots to compare model performance, prediction confidence, and model agreement.
- **Final Report:** Outputs a clear, final table summarizing the classification results for the unseen samples, including the predicted class and confidence score from the most confident model.

## Dataset
The project uses the `data_enose_unseen.xlsx` file, which contains two sheets:
- **`Sheet1` (Training Data):** Contains 130 labeled samples with readings from 14 e-nose sensor channels (`ch0` to `ch13`). The samples belong to three classes:
    - `WF`: 40 samples
    - `Ad`: 60 samples
    - `UF`: 30 samples
- **`unseen` (Testing Data):** Contains 10 unclassified samples (`X1` to `X10`) with corresponding readings from the 14 sensor channels. The goal of the script is to predict the class for these samples.

## Requirements
The script requires the following Python libraries. You can find them in the `requirements.txt` file.
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scipy`
- `openpyxl`

## Installation
1.  Clone the repository to your local machine:
    ```bash
    git clone https://github.com/ahmadfauzana/enose.git
    cd enose
    ```
2.  Install the required dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To run the complete analysis, simply execute the `main.py` script from your terminal:
```bash
python main.py
```
The script will automate the entire process:
1.  Load and preprocess the data.
2.  Perform and display plots for the Exploratory Data Analysis.
3.  Train, validate, and tune all machine learning models, printing performance metrics to the console.
4.  Predict the classes for the 10 unclassified samples.
5.  Display comprehensive visualizations comparing the prediction results.
6.  Print a final, detailed summary and classification table.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
