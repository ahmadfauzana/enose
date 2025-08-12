import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
import warnings
import os
import sys
from datetime import datetime
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create results directory
RESULTS_DIR = "enose_analysis_results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Create subdirectories
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
DATA_DIR = os.path.join(RESULTS_DIR, "data")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

for dir_path in [PLOTS_DIR, DATA_DIR, LOGS_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Set up logging to file
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

# Start logging
def setup_logging():
    """Set up logging with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOGS_DIR, f"analysis_log_{timestamp}.txt")
    logger = Logger(log_filename)
    return logger, timestamp

# Global variables will be set in main
timestamp = None
logger = None

def load_and_preprocess_data():
    """Load and preprocess the e-nose dataset"""
    print("Loading and preprocessing data...")
    
    # Load training data (Sheet1)
    train_data = pd.read_excel('data_enose_unseen.xlsx', sheet_name='Sheet1')
    
    # Load testing data (unseen) - these are unclassified cocoa bean samples
    test_data = pd.read_excel('data_enose_unseen.xlsx', sheet_name='unseen')
    
    # Clean column names and handle the unnamed first column
    train_data.columns = ['class'] + [f'ch{i}' for i in range(14)]
    test_data.columns = ['sample_id'] + [f'ch{i}' for i in range(14)]  # X1-X10 are sample IDs, not classes
    
    # Remove any rows with missing values
    train_data = train_data.dropna()
    test_data = test_data.dropna()
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    print(f"Training classes: {train_data['class'].unique()}")
    print(f"Training class distribution:")
    print(train_data['class'].value_counts())
    print(f"Unclassified samples to predict: {test_data['sample_id'].tolist()}")
    
    return train_data, test_data

def exploratory_data_analysis(train_data, test_data):
    """Perform comprehensive EDA with individual plots"""
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    # Basic statistics
    print("\nTraining Data Statistics:")
    print(train_data.describe())
    
    feature_cols = [f'ch{i}' for i in range(14)]
    
    # 1. Class distribution in training data
    plt.figure(figsize=(10, 6))
    train_data['class'].value_counts().plot(kind='bar', color=['red', 'blue', 'green'])
    plt.title('Training Data Class Distribution', fontsize=14)
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    for i, v in enumerate(train_data['class'].value_counts().values):
        plt.text(i, v + 1, str(v), ha='center', va='bottom')
    plt.tight_layout()
    
    plot1_file = os.path.join(PLOTS_DIR, f"01_class_distribution_{timestamp}.png")
    plt.savefig(plot1_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Class distribution plot saved to: {plot1_file}")
    plt.close()
    
    # 2. Sample distribution in testing data
    plt.figure(figsize=(10, 6))
    sample_counts = test_data['sample_id'].value_counts()
    sample_counts.plot(kind='bar', color='skyblue')
    plt.title('Testing Data Sample Distribution', fontsize=14)
    plt.xlabel('Sample IDs')
    plt.ylabel('Count (should all be 1)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot2_file = os.path.join(PLOTS_DIR, f"02_sample_distribution_{timestamp}.png")
    plt.savefig(plot2_file, dpi=300, bbox_inches='tight')
    print(f"✅ Sample distribution plot saved to: {plot2_file}")
    plt.close()
    
    # 3. Feature correlation heatmap
    plt.figure(figsize=(12, 10))
    corr_matrix = train_data[feature_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix', fontsize=14)
    plt.tight_layout()
    
    plot3_file = os.path.join(PLOTS_DIR, f"03_correlation_matrix_{timestamp}.png")
    plt.savefig(plot3_file, dpi=300, bbox_inches='tight')
    print(f"✅ Correlation matrix plot saved to: {plot3_file}")
    plt.close()
    
    # 4. Distribution of sensor readings (boxplot)
    plt.figure(figsize=(15, 8))
    train_data[feature_cols].boxplot()
    plt.title('Sensor Readings Distribution (Training Data)', fontsize=14)
    plt.xlabel('Sensor Channels')
    plt.ylabel('Sensor Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot4_file = os.path.join(PLOTS_DIR, f"04_sensor_boxplot_{timestamp}.png")
    plt.savefig(plot4_file, dpi=300, bbox_inches='tight')
    print(f"✅ Sensor boxplot saved to: {plot4_file}")
    plt.close()
    
    # 5. Feature importance using Random Forest
    plt.figure(figsize=(10, 8))
    le = LabelEncoder()
    y_encoded = le.fit_transform(train_data['class'])
    
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_temp.fit(train_data[feature_cols], y_encoded)
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_temp.feature_importances_
    }).sort_values('importance', ascending=True)
    
    feature_importance.plot(x='feature', y='importance', kind='barh', color='coral')
    plt.title('Feature Importance (Random Forest)', fontsize=14)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    
    plot5_file = os.path.join(PLOTS_DIR, f"05_feature_importance_{timestamp}.png")
    plt.savefig(plot5_file, dpi=300, bbox_inches='tight')
    print(f"✅ Feature importance plot saved to: {plot5_file}")
    plt.close()
    
    # 6. PCA visualization with class labels
    plt.figure(figsize=(10, 8))
    from sklearn.decomposition import PCA
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_data[feature_cols])
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create color map for classes
    unique_classes = train_data['class'].unique()
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, class_name in enumerate(unique_classes):
        mask = train_data['class'] == class_name
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=colors[i], label=class_name, alpha=0.7, s=50)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.title('PCA Visualization (Training Data)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot6_file = os.path.join(PLOTS_DIR, f"06_pca_visualization_{timestamp}.png")
    plt.savefig(plot6_file, dpi=300, bbox_inches='tight')
    print(f"✅ PCA visualization plot saved to: {plot6_file}")
    plt.close()
    
    # 7. Class-wise sensor means comparison
    plt.figure(figsize=(14, 8))
    class_means = train_data.groupby('class')[feature_cols].mean()
    
    x = np.arange(len(feature_cols))
    width = 0.25
    
    for i, class_name in enumerate(class_means.index):
        plt.bar(x + i*width, class_means.loc[class_name], width, 
                label=class_name, alpha=0.8)
    
    plt.xlabel('Sensor Channels')
    plt.ylabel('Mean Sensor Values')
    plt.title('Mean Sensor Values by Class')
    plt.xticks(x + width, feature_cols)
    plt.legend()
    plt.tight_layout()
    
    plot7_file = os.path.join(PLOTS_DIR, f"07_class_means_comparison_{timestamp}.png")
    plt.savefig(plot7_file, dpi=300, bbox_inches='tight')
    print(f"✅ Class means comparison plot saved to: {plot7_file}")
    plt.close()
    
    # Additional analysis: Class-wise feature means (only for training data)
    print("\nTraining Data Class-wise Feature Analysis:")
    print("="*40)
    class_means = train_data.groupby('class')[feature_cols].mean()
    print("\nMean sensor values by class (training data):")
    print(class_means)
    
    # Save class means to CSV
    class_means_file = os.path.join(DATA_DIR, f"class_means_{timestamp}.csv")
    class_means.to_csv(class_means_file)
    print(f"\n✅ Class means saved to: {class_means_file}")
    
    # Statistical significance test between classes (training data only)
    from scipy import stats
    print("\nClass separability analysis (ANOVA F-statistic for each feature):")
    classes = train_data['class'].unique()
    
    anova_results = []
    for feature in feature_cols:
        class_data = [train_data[train_data['class'] == cls][feature].values for cls in classes]
        f_stat, p_value = stats.f_oneway(*class_data)
        print(f"{feature}: F={f_stat:.2f}, p={p_value:.2e}")
        anova_results.append({
            'feature': feature,
            'f_statistic': f_stat,
            'p_value': p_value
        })
    
    # Save ANOVA results
    anova_df = pd.DataFrame(anova_results)
    anova_file = os.path.join(DATA_DIR, f"anova_results_{timestamp}.csv")
    anova_df.to_csv(anova_file, index=False)
    print(f"\n✅ ANOVA results saved to: {anova_file}")
    
    # Compare feature distributions between training and test data
    print("\nFeature Distribution Comparison (Training vs Unclassified Samples):")
    print("="*60)
    print(f"{'Feature':<8} {'Train Mean':<12} {'Test Mean':<12} {'Train Std':<12} {'Test Std':<12}")
    print("-" * 60)
    
    feature_comparison = []
    for feature in feature_cols:
        train_vals = train_data[feature].values
        test_vals = test_data[feature].values
        
        train_mean = np.mean(train_vals)
        test_mean = np.mean(test_vals)
        train_std = np.std(train_vals)
        test_std = np.std(test_vals)
        
        print(f"{feature:<8} {train_mean:<12.2f} {test_mean:<12.2f} {train_std:<12.2f} {test_std:<12.2f}")
        
        feature_comparison.append({
            'feature': feature,
            'train_mean': train_mean,
            'test_mean': test_mean,
            'train_std': train_std,
            'test_std': test_std
        })
    
    # Save feature comparison
    comparison_df = pd.DataFrame(feature_comparison)
    comparison_file = os.path.join(DATA_DIR, f"feature_comparison_{timestamp}.csv")
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n✅ Feature comparison saved to: {comparison_file}")
    
    return feature_cols

def prepare_data_for_modeling(train_data, test_data, feature_cols):
    """Prepare data for machine learning models"""
    print("\n" + "="*50)
    print("DATA PREPARATION")
    print("="*50)
    
    # Separate features and targets for training
    X_train = train_data[feature_cols]
    y_train = train_data['class']
    
    # Test data features (no labels - these are what we want to predict)
    X_test = test_data[feature_cols]
    test_sample_ids = test_data['sample_id']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create validation split from training data (stratified)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train, test_size=0.3, random_state=42, 
        stratify=y_train if len(y_train.unique()) > 1 else None
    )
    
    print(f"Training set size: {X_train_split.shape}")
    print(f"Validation set size: {X_val_split.shape}")
    print(f"Unclassified samples to predict: {X_test_scaled.shape}")
    print(f"Sample IDs: {test_sample_ids.tolist()}")
    
    return X_train_scaled, X_test_scaled, y_train, test_sample_ids, X_train_split, X_val_split, y_train_split, y_val_split, scaler

def train_multiple_models(X_train, X_val, y_train, y_val):
    """Train multiple machine learning models"""
    print("\n" + "="*50)
    print("MODEL TRAINING AND EVALUATION")
    print("="*50)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': SVC(random_state=42, probability=True),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Neural Network': MLPClassifier(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    # Train models and store results
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Validate on validation set
        val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='weighted')
        
        # Cross-validation on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'validation_accuracy': val_accuracy,
            'validation_f1': val_f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model': model
        }
        
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation F1-Score: {val_f1:.4f}")
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save model performance results
    performance_data = []
    for name, result in results.items():
        performance_data.append({
            'model': name,
            'validation_accuracy': result['validation_accuracy'],
            'validation_f1': result['validation_f1'],
            'cv_mean': result['cv_mean'],
            'cv_std': result['cv_std']
        })
    
    performance_df = pd.DataFrame(performance_data)
    performance_file = os.path.join(DATA_DIR, f"model_performance_{timestamp}.csv")
    performance_df.to_csv(performance_file, index=False)
    print(f"\n✅ Model performance results saved to: {performance_file}")
    
    return results, trained_models

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for best models"""
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING")
    print("="*50)
    
    # Parameter grids for top models
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        },
        'Support Vector Machine': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    }
    
    tuned_models = {}
    
    for model_name, param_grid in param_grids.items():
        print(f"\nTuning {model_name}...")
        
        if model_name == 'Random Forest':
            base_model = RandomForestClassifier(random_state=42)
        elif model_name == 'Support Vector Machine':
            base_model = SVC(random_state=42, probability=True)
        elif model_name == 'Gradient Boosting':
            base_model = GradientBoostingClassifier(random_state=42)
        
        grid_search = GridSearchCV(
            base_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        tuned_models[model_name] = {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Save hyperparameter tuning results
    tuning_data = []
    for name, result in tuned_models.items():
        tuning_data.append({
            'model': name,
            'best_score': result['best_score'],
            'best_params': str(result['best_params'])
        })
    
    tuning_df = pd.DataFrame(tuning_data)
    tuning_file = os.path.join(DATA_DIR, f"hyperparameter_tuning_{timestamp}.csv")
    tuning_df.to_csv(tuning_file, index=False)
    print(f"\n✅ Hyperparameter tuning results saved to: {tuning_file}")
    
    return tuned_models

def predict_unclassified_samples(models, tuned_models, X_test, test_sample_ids):
    """Make predictions for unclassified samples"""
    print("\n" + "="*50)
    print("PREDICTING UNCLASSIFIED SAMPLES")
    print("="*50)
    
    all_predictions = {}
    prediction_probabilities = {}
    
    # Predict with original models
    for name, model_info in models.items():
        model = model_info['model']
        y_pred = model.predict(X_test)
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
            max_probabilities = np.max(y_pred_proba, axis=1)
            avg_confidence = np.mean(max_probabilities)
            prediction_probabilities[name] = y_pred_proba
        else:
            max_probabilities = None
            avg_confidence = None
        
        all_predictions[name] = {
            'predictions': y_pred,
            'probabilities': max_probabilities,
            'avg_confidence': avg_confidence,
            'type': 'original'
        }
        
        print(f"\n{name}:")
        print(f"Average Prediction Confidence: {avg_confidence:.4f}" if avg_confidence else "N/A")
        
        # Show predictions for each sample
        for sample_id, pred, prob in zip(test_sample_ids, y_pred, max_probabilities if max_probabilities is not None else [None]*len(y_pred)):
            prob_str = f" (confidence: {prob:.3f})" if prob is not None else ""
            print(f"  {sample_id}: {pred}{prob_str}")
    
    # Predict with tuned models
    for name, model_info in tuned_models.items():
        model = model_info['model']
        y_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
            max_probabilities = np.max(y_pred_proba, axis=1)
            avg_confidence = np.mean(max_probabilities)
            prediction_probabilities[f"{name} (Tuned)"] = y_pred_proba
        else:
            max_probabilities = None
            avg_confidence = None
        
        all_predictions[f"{name} (Tuned)"] = {
            'predictions': y_pred,
            'probabilities': max_probabilities,
            'avg_confidence': avg_confidence,
            'type': 'tuned'
        }
        
        print(f"\n{name} (Tuned):")
        print(f"Average Prediction Confidence: {avg_confidence:.4f}" if avg_confidence else "N/A")
        
        # Show predictions for each sample
        for sample_id, pred, prob in zip(test_sample_ids, y_pred, max_probabilities if max_probabilities is not None else [None]*len(y_pred)):
            prob_str = f" (confidence: {prob:.3f})" if prob is not None else ""
            print(f"  {sample_id}: {pred}{prob_str}")
    
    # Save all predictions to comprehensive CSV
    all_predictions_data = []
    
    for model_name, pred_info in all_predictions.items():
        for i, sample_id in enumerate(test_sample_ids):
            prediction = pred_info['predictions'][i]
            confidence = pred_info['probabilities'][i] if pred_info['probabilities'] is not None else None
            
            all_predictions_data.append({
                'sample_id': sample_id,
                'model': model_name,
                'predicted_class': prediction,
                'confidence': confidence,
                'model_type': pred_info['type']
            })
    
    predictions_df = pd.DataFrame(all_predictions_data)
    predictions_file = os.path.join(DATA_DIR, f"all_predictions_{timestamp}.csv")
    predictions_df.to_csv(predictions_file, index=False)
    print(f"\n✅ All predictions saved to: {predictions_file}")
    
    return all_predictions, prediction_probabilities

def create_prediction_visualizations(all_predictions, prediction_probabilities, test_sample_ids):
    """Create comprehensive visualizations for predictions as individual plots"""
    print("\n" + "="*50)
    print("CREATING PREDICTION VISUALIZATIONS")
    print("="*50)
    
    model_names = list(all_predictions.keys())
    
    # 1. Model confidence comparison
    plt.figure(figsize=(14, 8))
    confidences = [all_predictions[name].get('avg_confidence', 0) or 0 for name in model_names]
    
    colors = ['skyblue' if 'Tuned' not in name else 'orange' for name in model_names]
    bars = plt.bar(range(len(model_names)), confidences, color=colors)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Average Prediction Confidence', fontsize=12)
    plt.title('Model Confidence Comparison', fontsize=14)
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    
    # Add confidence values on bars
    for bar, conf in zip(bars, confidences):
        if conf > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{conf:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    confidence_plot_file = os.path.join(PLOTS_DIR, f"08_model_confidence_{timestamp}.png")
    plt.savefig(confidence_plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Model confidence plot saved to: {confidence_plot_file}")
    plt.close()
    
    # 2. Prediction distribution across classes
    plt.figure(figsize=(14, 8))
    all_model_predictions = {}
    for name, results in all_predictions.items():
        pred_counts = np.unique(results['predictions'], return_counts=True)
        all_model_predictions[name] = dict(zip(pred_counts[0], pred_counts[1]))
    
    training_classes = ['WF', 'Ad', 'UF']
    x_pos = np.arange(len(model_names))
    bottom = np.zeros(len(model_names))
    
    colors_classes = ['red', 'blue', 'green']
    for i, cls in enumerate(training_classes):
        values = [all_model_predictions.get(name, {}).get(cls, 0) for name in model_names]
        plt.bar(x_pos, values, bottom=bottom, label=cls, color=colors_classes[i], alpha=0.7)
        bottom += values
    
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Number of Predictions', fontsize=12)
    plt.title('Prediction Distribution Across Classes', fontsize=14)
    plt.xticks(x_pos, model_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    pred_dist_file = os.path.join(PLOTS_DIR, f"09_prediction_distribution_{timestamp}.png")
    plt.savefig(pred_dist_file, dpi=300, bbox_inches='tight')
    print(f"✅ Prediction distribution plot saved to: {pred_dist_file}")
    plt.close()
    
    # 3. Model agreement analysis
    plt.figure(figsize=(10, 6))
    sample_agreements = []
    for i in range(len(test_sample_ids)):
        sample_predictions = [all_predictions[name]['predictions'][i] for name in model_names]
        unique_preds, counts = np.unique(sample_predictions, return_counts=True)
        max_agreement = np.max(counts)
        sample_agreements.append(max_agreement)
    
    agreement_counts = np.bincount(sample_agreements)[1:]  # Exclude 0 agreements
    x_agreement = np.arange(1, len(agreement_counts) + 1)
    
    plt.bar(x_agreement, agreement_counts, color='lightcoral')
    plt.xlabel('Number of Models in Agreement', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Model Agreement Distribution', fontsize=14)
    plt.xticks(x_agreement)
    
    # Add values on bars
    for i, v in enumerate(agreement_counts):
        if v > 0:
            plt.text(i + 1, v + 0.1, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    agreement_file = os.path.join(PLOTS_DIR, f"10_model_agreement_{timestamp}.png")
    plt.savefig(agreement_file, dpi=300, bbox_inches='tight')
    print(f"✅ Model agreement plot saved to: {agreement_file}")
    plt.close()
    
    # 4. Individual sample predictions heatmap
    plt.figure(figsize=(14, 8))
    prediction_matrix = np.zeros((len(test_sample_ids), len(model_names)))
    class_to_num = {'WF': 0, 'Ad': 1, 'UF': 2}
    
    for j, model_name in enumerate(model_names):
        for i, pred in enumerate(all_predictions[model_name]['predictions']):
            prediction_matrix[i, j] = class_to_num.get(pred, -1)
    
    sns.heatmap(prediction_matrix, 
                xticklabels=[name.replace(' (Tuned)', '(T)') for name in model_names],
                yticklabels=test_sample_ids,
                cmap='viridis', 
                cbar_kws={'label': 'Predicted Class (0=WF, 1=Ad, 2=UF)'},
                annot=True, fmt='.0f')
    plt.title('Prediction Heatmap', fontsize=14)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Samples', fontsize=12)
    plt.tight_layout()
    
    heatmap_file = os.path.join(PLOTS_DIR, f"11_prediction_heatmap_{timestamp}.png")
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    print(f"✅ Prediction heatmap saved to: {heatmap_file}")
    plt.close()
    
    # 5. Confidence distributions for top 3 models
    top_3_models = sorted(all_predictions.items(), 
                         key=lambda x: x[1].get('avg_confidence', 0) or 0, 
                         reverse=True)[:3]
    
    for idx, (model_name, results) in enumerate(top_3_models):
        plt.figure(figsize=(10, 6))
        if results['probabilities'] is not None:
            plt.hist(results['probabilities'], bins=10, alpha=0.7, edgecolor='black', color='lightblue')
            plt.xlabel('Prediction Confidence', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title(f'{model_name} - Confidence Distribution', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Add statistics
            mean_conf = np.mean(results['probabilities'])
            std_conf = np.std(results['probabilities'])
            plt.axvline(mean_conf, color='red', linestyle='--', 
                       label=f'Mean: {mean_conf:.3f}')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No probability\navailable', 
                    ha='center', va='center', transform=plt.gca().transAxes,
                    fontsize=16)
            plt.title(f'{model_name} - No Probabilities Available', fontsize=14)
        
        plt.tight_layout()
        conf_dist_file = os.path.join(PLOTS_DIR, f"12_confidence_dist_{idx+1}_{model_name.replace(' ', '_')}_{timestamp}.png")
        plt.savefig(conf_dist_file, dpi=300, bbox_inches='tight')
        print(f"✅ Confidence distribution plot saved to: {conf_dist_file}")
        plt.close()
    
    # 6. Per-sample confidence (best model)
    best_model = max(all_predictions.items(), key=lambda x: x[1].get('avg_confidence', 0) or 0)
    
    plt.figure(figsize=(12, 6))
    if best_model[1]['probabilities'] is not None:
        x_samples = range(len(test_sample_ids))
        bars = plt.bar(x_samples, best_model[1]['probabilities'], color='lightgreen')
        plt.xlabel('Sample ID', fontsize=12)
        plt.ylabel('Prediction Confidence', fontsize=12)
        plt.title(f'Per-Sample Confidence ({best_model[0]})', fontsize=14)
        plt.xticks(x_samples, test_sample_ids, rotation=45)
        
        # Add confidence values on bars
        for bar, conf in zip(bars, best_model[1]['probabilities']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{conf:.3f}', ha='center', va='bottom', rotation=45)
        
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No confidence scores available', 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=16)
    
    plt.tight_layout()
    sample_conf_file = os.path.join(PLOTS_DIR, f"13_per_sample_confidence_{timestamp}.png")
    plt.savefig(sample_conf_file, dpi=300, bbox_inches='tight')
    print(f"✅ Per-sample confidence plot saved to: {sample_conf_file}")
    plt.close()
    
    # 7. Consensus predictions visualization
    plt.figure(figsize=(12, 8))
    consensus_data = []
    consensus_predictions = []
    agreement_levels = []
    
    for i, sample_id in enumerate(test_sample_ids):
        sample_predictions = [all_predictions[name]['predictions'][i] for name in model_names]
        unique_preds, counts = np.unique(sample_predictions, return_counts=True)
        
        if len(unique_preds) == 1:
            consensus = unique_preds[0]
            agreement_count = len(model_names)
        else:
            max_count_idx = np.argmax(counts)
            consensus = unique_preds[max_count_idx]
            agreement_count = counts[max_count_idx]
        
        consensus_predictions.append(consensus)
        agreement_levels.append(agreement_count)
    
    # Create stacked bar chart
    x_pos = np.arange(len(test_sample_ids))
    colors_consensus = {'WF': 'red', 'Ad': 'blue', 'UF': 'green'}
    
    for i, (sample_id, pred, agreement) in enumerate(zip(test_sample_ids, consensus_predictions, agreement_levels)):
        color = colors_consensus.get(pred, 'gray')
        alpha = agreement / len(model_names)  # Transparency based on agreement level
        plt.bar(i, 1, color=color, alpha=alpha, edgecolor='black')
        plt.text(i, 0.5, f'{pred}\n{agreement}/{len(model_names)}', 
                ha='center', va='center', fontweight='bold')
    
    plt.xlabel('Sample ID', fontsize=12)
    plt.ylabel('Consensus', fontsize=12)
    plt.title('Consensus Predictions (Transparency = Agreement Level)', fontsize=14)
    plt.xticks(x_pos, test_sample_ids, rotation=45)
    
    # Add legend
    for class_name, color in colors_consensus.items():
        plt.bar([], [], color=color, label=class_name)
    plt.legend()
    
    plt.tight_layout()
    consensus_file = os.path.join(PLOTS_DIR, f"14_consensus_predictions_{timestamp}.png")
    plt.savefig(consensus_file, dpi=300, bbox_inches='tight')
    print(f"✅ Consensus predictions plot saved to: {consensus_file}")
    plt.close()
    
    # Detailed prediction summary
    print("\nDETAILED PREDICTION SUMMARY:")
    print("="*50)
    
    # Find the most confident model
    best_model = max(all_predictions.items(), key=lambda x: x[1].get('avg_confidence', 0) or 0)
    print(f"\nMost confident model: {best_model[0]}")
    print(f"Average confidence: {best_model[1]['avg_confidence']:.4f}")
    
    # Show consensus predictions
    print(f"\nCONSENSUS PREDICTIONS:")
    print("-" * 30)
    
    consensus_data = []
    for i, sample_id in enumerate(test_sample_ids):
        sample_predictions = [all_predictions[name]['predictions'][i] for name in model_names]
        unique_preds, counts = np.unique(sample_predictions, return_counts=True)
        
        if len(unique_preds) == 1:
            # All models agree
            consensus = unique_preds[0]
            agreement_level = "UNANIMOUS"
            agreement_count = len(model_names)
        else:
            # Find majority
            max_count_idx = np.argmax(counts)
            consensus = unique_preds[max_count_idx]
            agreement_count = counts[max_count_idx]
            agreement_level = f"{agreement_count}/{len(model_names)} models"
        
        print(f"{sample_id}: {consensus} ({agreement_level})")
        
        consensus_data.append({
            'sample_id': sample_id,
            'consensus_prediction': consensus,
            'agreement_level': agreement_level,
            'agreement_count': agreement_count,
            'total_models': len(model_names)
        })
    
    # Save consensus predictions
    consensus_df = pd.DataFrame(consensus_data)
    consensus_file = os.path.join(DATA_DIR, f"consensus_predictions_{timestamp}.csv")
    consensus_df.to_csv(consensus_file, index=False)
    print(f"\n✅ Consensus predictions saved to: {consensus_file}")
    
    return best_model

def main():
    """Main execution function"""
    global timestamp, logger
    
    # Set up logging
    logger, timestamp = setup_logging()
    sys.stdout = logger
    
    print("E-NOSE COCOA BEAN CLASSIFICATION WITH MULTIPLE ML MODELS")
    print("=" * 70)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results will be saved to: {RESULTS_DIR}")
    print("=" * 70)
    
    try:
        # Load and preprocess data
        train_data, test_data = load_and_preprocess_data()
        
        # Exploratory data analysis
        feature_cols = exploratory_data_analysis(train_data, test_data)
        
        # Prepare data for modeling
        X_train_full, X_test, y_train_full, test_sample_ids, X_train, X_val, y_train, y_val, scaler = prepare_data_for_modeling(
            train_data, test_data, feature_cols
        )
        
        # Train multiple models
        model_results, trained_models = train_multiple_models(X_train, X_val, y_train, y_val)
        
        # Hyperparameter tuning
        tuned_models = hyperparameter_tuning(X_train_full, y_train_full)
        
        # Make predictions on unclassified samples
        all_predictions, prediction_probabilities = predict_unclassified_samples(model_results, tuned_models, X_test, test_sample_ids)
        
        # Create prediction visualizations
        best_model = create_prediction_visualizations(all_predictions, prediction_probabilities, test_sample_ids)
        
        # Summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        # Find best model by confidence
        best_model_name = best_model[0]
        best_confidence = best_model[1].get('avg_confidence', 'N/A')
        print(f"\nMost confident model: {best_model_name}")
        print(f"Average Prediction Confidence: {best_confidence}")
        
        # Show final predictions from best model
        print(f"\nFINAL PREDICTIONS (using {best_model_name}):")
        print("-" * 40)
        predictions = best_model[1]['predictions']
        probabilities = best_model[1]['probabilities']
        
        for i, sample_id in enumerate(test_sample_ids):
            prob_str = f" (confidence: {probabilities[i]:.3f})" if probabilities is not None else ""
            print(f"{sample_id}: {predictions[i]}{prob_str}")
        
        # Prediction distribution
        pred_counts = np.unique(predictions, return_counts=True)
        print(f"\nPREDICTION SUMMARY:")
        total_samples = len(predictions)
        for pred_class, count in zip(pred_counts[0], pred_counts[1]):
            percentage = (count / total_samples) * 100
            print(f"- {pred_class}: {count}/{total_samples} samples ({percentage:.1f}%)")
        
        # Feature importance from Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_full, y_train_full)
        
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTOP 5 MOST IMPORTANT SENSOR CHANNELS:")
        for i, (_, row) in enumerate(feature_importance.head().iterrows()):
            print(f"{i+1}. {row['feature']}: {row['importance']:.4f}")
        
        print(f"\nDATASET CHARACTERISTICS:")
        print(f"- Training samples: {len(train_data)} (WF: 40, Ad: 60, UF: 30)")
        print(f"- Unclassified cocoa bean samples: {len(test_data)} (X1-X10)")
        print(f"- Sensor channels: {len(feature_cols)} (ch0-ch13)")
        print(f"- Training classes: {', '.join(train_data['class'].unique())}")
        
        # Create final prediction table
        print(f"\n" + "="*60)
        print("FINAL CLASSIFICATION RESULTS")
        print("="*60)
        
        results_df = pd.DataFrame({
            'Sample_ID': test_sample_ids,
            'Predicted_Class': predictions,
            'Confidence': probabilities if probabilities is not None else ['N/A'] * len(predictions),
            'Model_Used': [best_model_name] * len(predictions)
        })
        
        print(results_df.to_string(index=False))
        
        # Save final results to CSV
        final_results_file = os.path.join(DATA_DIR, f"final_classification_results_{timestamp}.csv")
        results_df.to_csv(final_results_file, index=False)
        print(f"\n✅ Final classification results saved to: {final_results_file}")
        
        print(f"\n" + "="*60)
        print("INTERPRETATION:")
        print("- WF, Ad, UF likely represent different types/qualities of cocoa beans")
        print("- X1-X10 are unclassified cocoa bean samples")
        print("- The model predicts which known category each sample belongs to")
        print("- Higher confidence scores indicate more reliable predictions")
        print("- Consider validating results with domain experts")
        print("="*60)
        
        # Create comprehensive summary report
        create_summary_report(timestamp, train_data, test_data, feature_cols, 
                             best_model, results_df, feature_importance)
        
        # Close logging
        print(f"\n" + "="*60)
        print("ANALYSIS COMPLETE - ALL RESULTS SAVED")
        print("="*60)
        print(f"📁 Results directory: {RESULTS_DIR}")
        print(f"📊 Individual plots (14 files) saved in: {PLOTS_DIR}")
        print(f"📋 Data files (8 CSV files) saved in: {DATA_DIR}")
        print(f"📝 Log files saved in: {LOGS_DIR}")
        print(f"📄 Summary report: ANALYSIS_SUMMARY_REPORT_{timestamp}.txt")
        print("="*60)
        print("\n🎉 All visualizations are now individual, high-quality files!")
        print("   Each plot can be used separately in presentations or publications.")
        
    except Exception as e:
        print(f"\n❌ Error occurred during analysis: {str(e)}")
        print("Check the log file for detailed error information.")
    
    finally:
        # Ensure logging is properly closed
        if 'logger' in globals():
            logger.close()
            sys.stdout = sys.__stdout__

def create_summary_report(timestamp, train_data, test_data, feature_cols, best_model, results_df, feature_importance):
    """Create a comprehensive summary report"""
    
    report_file = os.path.join(RESULTS_DIR, f"ANALYSIS_SUMMARY_REPORT_{timestamp}.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("E-NOSE COCOA BEAN CLASSIFICATION - COMPREHENSIVE ANALYSIS REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("="*80 + "\n\n")
        
        # Dataset Overview
        f.write("DATASET OVERVIEW:\n")
        f.write("-"*40 + "\n")
        f.write(f"Training Samples: {len(train_data)}\n")
        f.write(f"  - WF: {len(train_data[train_data['class'] == 'WF'])}\n")
        f.write(f"  - Ad: {len(train_data[train_data['class'] == 'Ad'])}\n")
        f.write(f"  - UF: {len(train_data[train_data['class'] == 'UF'])}\n")
        f.write(f"Unclassified Samples: {len(test_data)} (X1-X10)\n")
        f.write(f"Features: {len(feature_cols)} sensor channels (ch0-ch13)\n\n")
        
        # Best Model Information
        f.write("BEST PERFORMING MODEL:\n")
        f.write("-"*40 + "\n")
        f.write(f"Model: {best_model[0]}\n")
        f.write(f"Average Confidence: {best_model[1].get('avg_confidence', 'N/A')}\n\n")
        
        # Final Classifications
        f.write("FINAL CLASSIFICATION RESULTS:\n")
        f.write("-"*40 + "\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        
        # Prediction Summary
        predictions = results_df['Predicted_Class'].values
        pred_counts = np.unique(predictions, return_counts=True)
        f.write("PREDICTION DISTRIBUTION:\n")
        f.write("-"*40 + "\n")
        total_samples = len(predictions)
        for pred_class, count in zip(pred_counts[0], pred_counts[1]):
            percentage = (count / total_samples) * 100
            f.write(f"{pred_class}: {count}/{total_samples} samples ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Feature Importance
        f.write("TOP 10 MOST IMPORTANT SENSOR CHANNELS:\n")
        f.write("-"*40 + "\n")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            f.write(f"{i+1:2d}. {row['feature']}: {row['importance']:.4f}\n")
        f.write("\n")
        
        # Files Generated
        f.write("FILES GENERATED:\n")
        f.write("-"*40 + "\n")
        f.write("Individual Plots (High Resolution PNG):\n")
        f.write(f"  01. Class Distribution: 01_class_distribution_{timestamp}.png\n")
        f.write(f"  02. Sample Distribution: 02_sample_distribution_{timestamp}.png\n")
        f.write(f"  03. Correlation Matrix: 03_correlation_matrix_{timestamp}.png\n")
        f.write(f"  04. Sensor Boxplot: 04_sensor_boxplot_{timestamp}.png\n")
        f.write(f"  05. Feature Importance: 05_feature_importance_{timestamp}.png\n")
        f.write(f"  06. PCA Visualization: 06_pca_visualization_{timestamp}.png\n")
        f.write(f"  07. Class Means Comparison: 07_class_means_comparison_{timestamp}.png\n")
        f.write(f"  08. Model Confidence: 08_model_confidence_{timestamp}.png\n")
        f.write(f"  09. Prediction Distribution: 09_prediction_distribution_{timestamp}.png\n")
        f.write(f"  10. Model Agreement: 10_model_agreement_{timestamp}.png\n")
        f.write(f"  11. Prediction Heatmap: 11_prediction_heatmap_{timestamp}.png\n")
        f.write(f"  12-14. Confidence Distributions: 12_confidence_dist_*_{timestamp}.png\n")
        f.write(f"  13. Per-Sample Confidence: 13_per_sample_confidence_{timestamp}.png\n")
        f.write(f"  14. Consensus Predictions: 14_consensus_predictions_{timestamp}.png\n\n")
        
        f.write("Data Files:\n")
        f.write(f"  - final_classification_results_{timestamp}.csv\n")
        f.write(f"  - all_predictions_{timestamp}.csv\n")
        f.write(f"  - consensus_predictions_{timestamp}.csv\n")
        f.write(f"  - model_performance_{timestamp}.csv\n")
        f.write(f"  - hyperparameter_tuning_{timestamp}.csv\n")
        f.write(f"  - class_means_{timestamp}.csv\n")
        f.write(f"  - anova_results_{timestamp}.csv\n")
        f.write(f"  - feature_comparison_{timestamp}.csv\n\n")
        
        f.write("Log Files:\n")
        f.write(f"  - analysis_log_{timestamp}.txt\n\n")
        
        # Interpretation and Recommendations
        f.write("INTERPRETATION & RECOMMENDATIONS:\n")
        f.write("-"*40 + "\n")
        f.write("1. WF, Ad, UF represent different cocoa bean types/qualities\n")
        f.write("2. X1-X10 samples have been classified into these categories\n")
        f.write("3. Higher confidence scores indicate more reliable predictions\n")
        f.write("4. Review samples with low confidence scores manually\n")
        f.write("5. Validate results with domain experts\n")
        f.write("6. Consider the top sensor channels for future measurements\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"\n✅ Comprehensive summary report saved to: {report_file}")

if __name__ == "__main__":
    main()