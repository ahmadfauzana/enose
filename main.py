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
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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
    """Perform comprehensive EDA"""
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    # Basic statistics
    print("\nTraining Data Statistics:")
    print(train_data.describe())
    
    # Class distribution
    plt.figure(figsize=(15, 10))
    
    # Class distribution in training data
    plt.subplot(2, 3, 1)
    train_data['class'].value_counts().plot(kind='bar')
    plt.title('Training Data Class Distribution')
    plt.xticks(rotation=45)
    
    # Sample distribution in testing data (just show sample IDs)
    plt.subplot(2, 3, 2)
    sample_counts = test_data['sample_id'].value_counts()
    sample_counts.plot(kind='bar')
    plt.title('Testing Data Sample Distribution')
    plt.ylabel('Count (should all be 1)')
    plt.xticks(rotation=45)
    
    # Feature correlation heatmap
    plt.subplot(2, 3, 3)
    feature_cols = [f'ch{i}' for i in range(14)]
    corr_matrix = train_data[feature_cols].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    
    # Distribution of sensor readings
    plt.subplot(2, 3, 4)
    train_data[feature_cols].boxplot()
    plt.title('Sensor Readings Distribution (Training)')
    plt.xticks(rotation=45)
    
    # Feature importance using Random Forest
    plt.subplot(2, 3, 5)
    # Encode labels for feature importance calculation
    le = LabelEncoder()
    y_encoded = le.fit_transform(train_data['class'])
    
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_temp.fit(train_data[feature_cols], y_encoded)
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_temp.feature_importances_
    }).sort_values('importance', ascending=True)
    
    feature_importance.plot(x='feature', y='importance', kind='barh')
    plt.title('Feature Importance (Random Forest)')
    
    # PCA visualization with class labels
    from sklearn.decomposition import PCA
    plt.subplot(2, 3, 6)
    
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
                   c=colors[i], label=class_name, alpha=0.7)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA Visualization (Training Data)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis: Class-wise feature means (only for training data)
    print("\nTraining Data Class-wise Feature Analysis:")
    print("="*40)
    class_means = train_data.groupby('class')[feature_cols].mean()
    print("\nMean sensor values by class (training data):")
    print(class_means)
    
    # Statistical significance test between classes (training data only)
    from scipy import stats
    print("\nClass separability analysis (ANOVA F-statistic for each feature):")
    classes = train_data['class'].unique()
    for feature in feature_cols:
        class_data = [train_data[train_data['class'] == cls][feature].values for cls in classes]
        f_stat, p_value = stats.f_oneway(*class_data)
        print(f"{feature}: F={f_stat:.2f}, p={p_value:.2e}")
    
    # Compare feature distributions between training and test data
    print("\nFeature Distribution Comparison (Training vs Unclassified Samples):")
    print("="*60)
    print(f"{'Feature':<8} {'Train Mean':<12} {'Test Mean':<12} {'Train Std':<12} {'Test Std':<12}")
    print("-" * 60)
    
    for feature in feature_cols:
        train_vals = train_data[feature].values
        test_vals = test_data[feature].values
        
        train_mean = np.mean(train_vals)
        test_mean = np.mean(test_vals)
        train_std = np.std(train_vals)
        test_std = np.std(test_vals)
        
        print(f"{feature:<8} {train_mean:<12.2f} {test_mean:<12.2f} {train_std:<12.2f} {test_std:<12.2f}")
    
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
        X_train_scaled, y_train, test_size=0.2, random_state=42, 
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
    
    return all_predictions, prediction_probabilities

def create_prediction_visualizations(all_predictions, prediction_probabilities, test_sample_ids):
    """Create comprehensive visualizations for predictions"""
    print("\n" + "="*50)
    print("CREATING PREDICTION VISUALIZATIONS")
    print("="*50)
    
    plt.figure(figsize=(20, 15))
    
    # Model confidence comparison
    plt.subplot(3, 3, 1)
    model_names = list(all_predictions.keys())
    confidences = [all_predictions[name].get('avg_confidence', 0) or 0 for name in model_names]
    
    colors = ['skyblue' if 'Tuned' not in name else 'orange' for name in model_names]
    bars = plt.bar(range(len(model_names)), confidences, color=colors)
    plt.xlabel('Models')
    plt.ylabel('Average Prediction Confidence')
    plt.title('Model Confidence Comparison')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    
    # Add confidence values on bars
    for bar, conf in zip(bars, confidences):
        if conf > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{conf:.3f}', ha='center', va='bottom')
    
    # Prediction distribution across all models
    plt.subplot(3, 3, 2)
    all_model_predictions = {}
    for name, results in all_predictions.items():
        pred_counts = np.unique(results['predictions'], return_counts=True)
        all_model_predictions[name] = dict(zip(pred_counts[0], pred_counts[1]))
    
    # Create a stacked bar chart
    training_classes = ['WF', 'Ad', 'UF']
    x_pos = np.arange(len(model_names))
    bottom = np.zeros(len(model_names))
    
    colors_classes = ['red', 'blue', 'green']
    for i, cls in enumerate(training_classes):
        values = [all_model_predictions.get(name, {}).get(cls, 0) for name in model_names]
        plt.bar(x_pos, values, bottom=bottom, label=cls, color=colors_classes[i], alpha=0.7)
        bottom += values
    
    plt.xlabel('Models')
    plt.ylabel('Number of Predictions')
    plt.title('Prediction Distribution Across Classes')
    plt.xticks(x_pos, model_names, rotation=45, ha='right')
    plt.legend()
    
    # Model agreement analysis
    plt.subplot(3, 3, 3)
    # For each sample, count how many models agree on the prediction
    sample_agreements = []
    for i in range(len(test_sample_ids)):
        sample_predictions = [all_predictions[name]['predictions'][i] for name in model_names]
        # Count the most frequent prediction
        unique_preds, counts = np.unique(sample_predictions, return_counts=True)
        max_agreement = np.max(counts)
        sample_agreements.append(max_agreement)
    
    agreement_counts = np.bincount(sample_agreements)[1:]  # Exclude 0 agreements
    x_agreement = np.arange(1, len(agreement_counts) + 1)
    
    plt.bar(x_agreement, agreement_counts)
    plt.xlabel('Number of Models in Agreement')
    plt.ylabel('Number of Samples')
    plt.title('Model Agreement Distribution')
    plt.xticks(x_agreement)
    
    # Individual sample predictions heatmap
    plt.subplot(3, 3, 4)
    # Create a matrix: rows = samples, columns = models, values = predicted class
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
                annot=False)
    plt.title('Prediction Heatmap')
    plt.xlabel('Models')
    plt.ylabel('Samples')
    
    # Confidence distribution for best models
    top_3_models = sorted(all_predictions.items(), 
                         key=lambda x: x[1].get('avg_confidence', 0) or 0, 
                         reverse=True)[:3]
    
    for i, (model_name, results) in enumerate(top_3_models):
        plt.subplot(3, 3, 5 + i)
        if results['probabilities'] is not None:
            plt.hist(results['probabilities'], bins=10, alpha=0.7, edgecolor='black')
            plt.xlabel('Prediction Confidence')
            plt.ylabel('Frequency')
            plt.title(f'{model_name}\nConfidence Distribution')
        else:
            plt.text(0.5, 0.5, 'No probability\navailable', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'{model_name}\nNo Probabilities')
    
    # Sample-wise prediction confidence
    plt.subplot(3, 3, 8)
    best_model = top_3_models[0]
    if best_model[1]['probabilities'] is not None:
        x_samples = range(len(test_sample_ids))
        plt.bar(x_samples, best_model[1]['probabilities'])
        plt.xlabel('Sample Index')
        plt.ylabel('Prediction Confidence')
        plt.title(f'Per-Sample Confidence ({best_model[0]})')
        plt.xticks(x_samples, test_sample_ids, rotation=45)
    
    plt.tight_layout()
    plt.show()
    
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
    for i, sample_id in enumerate(test_sample_ids):
        sample_predictions = [all_predictions[name]['predictions'][i] for name in model_names]
        unique_preds, counts = np.unique(sample_predictions, return_counts=True)
        
        if len(unique_preds) == 1:
            # All models agree
            consensus = unique_preds[0]
            agreement_level = "UNANIMOUS"
        else:
            # Find majority
            max_count_idx = np.argmax(counts)
            consensus = unique_preds[max_count_idx]
            agreement_level = f"{counts[max_count_idx]}/{len(model_names)} models"
        
        print(f"{sample_id}: {consensus} ({agreement_level})")
    
    return best_model

def main():
    """Main execution function"""
    print("E-NOSE COCOA BEAN CLASSIFICATION WITH MULTIPLE ML MODELS")
    print("=" * 70)
    
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
    
    print(f"\n" + "="*60)
    print("INTERPRETATION:")
    print("- WF, Ad, UF likely represent different types/qualities of cocoa beans")
    print("- X1-X10 are unclassified cocoa bean samples")
    print("- The model predicts which known category each sample belongs to")
    print("- Higher confidence scores indicate more reliable predictions")
    print("- Consider validating results with domain experts")
    print("="*60)

if __name__ == "__main__":
    main()