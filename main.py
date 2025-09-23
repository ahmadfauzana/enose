import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, f1_score,
                           precision_score, recall_score, matthews_corrcoef)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings
import os
import sys
from datetime import datetime
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DeepMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)


class Conv1DNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B,1,14)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class LSTMNet(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Treat each feature as "time-step=1" sequence
        x = x.unsqueeze(1)  # (B,1,14)
        out, (h, c) = self.lstm(x)
        return self.fc(h[-1])

def compute_feature_importances(model, X_val, y_val):
    """Compute feature importances if supported"""
    if isinstance(model, nn.Module):
        print("⚠️ Skipping feature importance: PyTorch model not compatible with permutation_importance")
        return None
    try:
        result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
        return result.importances_mean
    except Exception as e:
        print(f"❌ Error computing feature importance: {e}")
        return None
    
def create_run_directory():
    """Create a new numbered directory for this analysis run"""
    base_name = "enose_run"
    counter = 1
    
    # Find the next available run number
    while True:
        run_dir = f"{base_name}_{counter:02d}"
        if not os.path.exists(run_dir):
            break
        counter += 1
    
    # Create the main run directory
    os.makedirs(run_dir)
    
    # Create subdirectories within the run directory
    plots_dir = os.path.join(run_dir, "plots")
    data_dir = os.path.join(run_dir, "data")
    logs_dir = os.path.join(run_dir, "logs")
    
    for dir_path in [plots_dir, data_dir, logs_dir]:
        os.makedirs(dir_path)
    
    return run_dir, plots_dir, data_dir, logs_dir

# Create run-specific directories
RESULTS_DIR, PLOTS_DIR, DATA_DIR, LOGS_DIR = create_run_directory()

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
    log_filename = os.path.join(LOGS_DIR, f"analysis_log.txt")
    logger = Logger(log_filename)
    return logger, timestamp

# Global variables will be set in main
timestamp = None
logger = None

def load_and_preprocess_data():
    """Load and preprocess the e-nose dataset"""
    print("Loading and preprocessing data...")
    
    # Load training data (Sheet1)
    train_data = pd.read_excel('data enose_unseen.xlsx', sheet_name='Sheet1')
    
    # Load testing data (unseen) - these are unclassified cocoa bean samples
    test_data = pd.read_excel('data enose_unseen.xlsx', sheet_name='unseen')
    
    # Clean column names and handle the unnamed first column
    train_data.columns = ['class'] + [f'ch{i}' for i in range(14)]
    test_data.columns = ['sample_id'] + [f'ch{i}' for i in range(14)]  # X1-X10 are sample IDs, not classes
    
    # Update class names: WF->WFB, UF->UFB, Ad->ADB
    class_mapping = {'WF': 'WFB', 'UF': 'UFB', 'Ad': 'ADB'}
    train_data['class'] = train_data['class'].map(class_mapping)
    
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
    
    plot1_file = os.path.join(PLOTS_DIR, "01_class_distribution.png")
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
    
    plot2_file = os.path.join(PLOTS_DIR, "02_sample_distribution.png")
    plt.savefig(plot2_file, dpi=300, bbox_inches='tight')
    print(f"✅ Sample distribution plot saved to: {plot2_file}")
    plt.close()
    
    # 3. Feature correlation heatmap
    plt.figure(figsize=(12, 10))
    corr_matrix = train_data[feature_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix', fontsize=14)
    plt.tight_layout()
    
    plot3_file = os.path.join(PLOTS_DIR, "03_correlation_matrix.png")
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
    
    plot4_file = os.path.join(PLOTS_DIR, "04_sensor_boxplot.png")
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
    
    plot5_file = os.path.join(PLOTS_DIR, "05_feature_importance.png")
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
    
    plot6_file = os.path.join(PLOTS_DIR, "06_pca_visualization.png")
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
    
    plot7_file = os.path.join(PLOTS_DIR, "07_class_means_comparison.png")
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
    class_means_file = os.path.join(DATA_DIR, "class_means.csv")
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
    anova_file = os.path.join(DATA_DIR, "anova_results.csv")
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
    comparison_file = os.path.join(DATA_DIR, "feature_comparison.csv")
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n✅ Feature comparison saved to: {comparison_file}")
    
    return feature_cols

def comprehensive_data_analysis(train_data, test_data, feature_cols):
    """Perform comprehensive analysis of factors that can influence results"""
    print("\n" + "="*60)
    print("COMPREHENSIVE DATA ANALYSIS")
    print("="*60)
    
    # 1. Feature distribution analysis by class
    print("\n1. FEATURE DISTRIBUTION ANALYSIS BY CLASS")
    print("-" * 50)
    
    plt.figure(figsize=(20, 15))
    
    # Create subplot for each feature showing distribution by class
    n_features = len(feature_cols)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    for i, feature in enumerate(feature_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        
        for class_name in train_data['class'].unique():
            class_data = train_data[train_data['class'] == class_name][feature]
            plt.hist(class_data, alpha=0.6, label=class_name, bins=15)
        
        plt.title(f'{feature} Distribution by Class', fontsize=10)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    dist_plot_file = os.path.join(PLOTS_DIR, "15_feature_distributions_by_class.png")
    plt.savefig(dist_plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Feature distributions by class saved to: {dist_plot_file}")
    plt.close()
    
    # 2. Statistical significance analysis (detailed)
    print("\n2. STATISTICAL SIGNIFICANCE ANALYSIS")
    print("-" * 50)
    
    statistical_results = []
    for feature in feature_cols:
        # ANOVA F-test
        class_data = [train_data[train_data['class'] == cls][feature].values 
                     for cls in train_data['class'].unique()]
        f_stat, p_value = stats.f_oneway(*class_data)
        
        # Kruskal-Wallis test (non-parametric alternative)
        h_stat, h_p_value = stats.kruskal(*class_data)
        
        # Effect size (eta-squared)
        ss_between = sum([len(group) * (np.mean(group) - np.mean(train_data[feature]))**2 
                         for group in class_data])
        ss_total = np.sum((train_data[feature] - np.mean(train_data[feature]))**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        statistical_results.append({
            'feature': feature,
            'f_statistic': f_stat,
            'f_p_value': p_value,
            'kruskal_h': h_stat,
            'kruskal_p': h_p_value,
            'eta_squared': eta_squared,
            'significance': 'High' if p_value < 0.001 else 'Medium' if p_value < 0.01 else 'Low' if p_value < 0.05 else 'None'
        })
        
        print(f"{feature}: F={f_stat:.2f}, p={p_value:.2e}, η²={eta_squared:.3f} ({statistical_results[-1]['significance']} significance)")
    
    # Save statistical results
    stats_df = pd.DataFrame(statistical_results)
    stats_file = os.path.join(DATA_DIR, "detailed_statistical_analysis.csv")
    stats_df.to_csv(stats_file, index=False)
    print(f"\n✅ Detailed statistical analysis saved to: {stats_file}")
    
    # 3. Outlier detection and analysis
    print("\n3. OUTLIER DETECTION AND ANALYSIS")
    print("-" * 50)
    
    # Z-score based outlier detection
    z_scores = np.abs(stats.zscore(train_data[feature_cols]))
    outlier_threshold = 3
    outliers_zscore = (z_scores > outlier_threshold).sum(axis=1)
    
    # IQR based outlier detection
    Q1 = train_data[feature_cols].quantile(0.25)
    Q3 = train_data[feature_cols].quantile(0.75)
    IQR = Q3 - Q1
    outliers_iqr = ((train_data[feature_cols] < (Q1 - 1.5 * IQR)) | 
                    (train_data[feature_cols] > (Q3 + 1.5 * IQR))).sum(axis=1)
    
    # Create outlier visualization
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.hist(outliers_zscore, bins=20, alpha=0.7, color='red', edgecolor='black')
    plt.title('Distribution of Z-Score Based Outliers per Sample')
    plt.xlabel('Number of Outlier Features per Sample')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.hist(outliers_iqr, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Distribution of IQR Based Outliers per Sample')
    plt.xlabel('Number of Outlier Features per Sample')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Outliers by class
    plt.subplot(2, 2, 3)
    for class_name in train_data['class'].unique():
        class_mask = train_data['class'] == class_name
        class_outliers = outliers_zscore[class_mask]
        plt.hist(class_outliers, alpha=0.6, label=f'{class_name}', bins=10)
    plt.title('Z-Score Outliers by Class')
    plt.xlabel('Number of Outlier Features')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Outlier features frequency
    outlier_features = (z_scores > outlier_threshold).sum(axis=0)
    plt.bar(feature_cols, outlier_features, color='orange', alpha=0.7)
    plt.title('Outlier Frequency by Feature')
    plt.xlabel('Features')
    plt.ylabel('Number of Outlier Samples')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    outlier_plot_file = os.path.join(PLOTS_DIR, "16_outlier_analysis.png")
    plt.savefig(outlier_plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Outlier analysis plot saved to: {outlier_plot_file}")
    plt.close()
    
    print(f"Z-score outliers (>3σ): {(outliers_zscore > 0).sum()} samples affected")
    print(f"IQR outliers: {(outliers_iqr > 0).sum()} samples affected")
    
    # Fix the array indexing issue
    outlier_features = (z_scores > outlier_threshold).sum(axis=0)
    most_problematic_indices = np.argsort(outlier_features)[-3:][::-1]
    most_problematic_features = [feature_cols[i] for i in most_problematic_indices]
    print(f"Most problematic features (Z-score): {most_problematic_features}")
    
    # 4. Data quality assessment
    print("\n4. DATA QUALITY ASSESSMENT")
    print("-" * 50)
    
    quality_metrics = {}
    
    # Missing values
    missing_train = train_data[feature_cols].isnull().sum()
    missing_test = test_data[feature_cols].isnull().sum()
    
    # Variance analysis
    feature_variances = train_data[feature_cols].var()
    low_variance_features = feature_variances[feature_variances < feature_variances.quantile(0.1)]
    
    # Skewness analysis
    feature_skewness = train_data[feature_cols].skew()
    highly_skewed = feature_skewness[np.abs(feature_skewness) > 2]
    
    # Range analysis
    feature_ranges = train_data[feature_cols].max() - train_data[feature_cols].min()
    
    # Coefficient of variation (handle division by zero)
    feature_means = train_data[feature_cols].mean()
    feature_stds = train_data[feature_cols].std()
    cv_values = np.where(feature_means != 0, feature_stds / feature_means, 0)
    
    quality_metrics = {
        'feature': feature_cols,
        'missing_train': missing_train.values,
        'missing_test': missing_test.values,
        'variance': feature_variances.values,
        'skewness': feature_skewness.values,
        'range': feature_ranges.values,
        'cv': cv_values
    }
    
    quality_df = pd.DataFrame(quality_metrics)
    quality_file = os.path.join(DATA_DIR, "data_quality_metrics.csv")
    quality_df.to_csv(quality_file, index=False)
    print(f"✅ Data quality metrics saved to: {quality_file}")
    
    print(f"Low variance features: {len(low_variance_features)}")
    print(f"Highly skewed features (|skew| > 2): {len(highly_skewed)}")
    if len(highly_skewed) > 0:
        print(f"  Most skewed: {highly_skewed.abs().sort_values(ascending=False).head(3).index.tolist()}")
    
    # 5. Class separability analysis
    print("\n5. CLASS SEPARABILITY ANALYSIS")
    print("-" * 50)
    
    # Calculate class separability metrics
    class_means = train_data.groupby('class')[feature_cols].mean()
    class_stds = train_data.groupby('class')[feature_cols].std()
    
    # Calculate separability index for each feature
    separability_scores = []
    for feature in feature_cols:
        classes = train_data['class'].unique()
        max_separation = 0
        
        for i in range(len(classes)):
            for j in range(i+1, len(classes)):
                class1_data = train_data[train_data['class'] == classes[i]][feature]
                class2_data = train_data[train_data['class'] == classes[j]][feature]
                
                # Fisher's discriminant ratio
                mean_diff = abs(class1_data.mean() - class2_data.mean())
                var_sum = class1_data.var() + class2_data.var()
                
                if var_sum > 0:
                    fisher_ratio = (mean_diff ** 2) / var_sum
                    max_separation = max(max_separation, fisher_ratio)
        
        separability_scores.append(max_separation)
    
    # Visualize separability
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    bars = plt.bar(feature_cols, separability_scores, color='purple', alpha=0.7)
    plt.title('Feature Class Separability (Fisher Discriminant Ratio)')
    plt.xlabel('Features')
    plt.ylabel('Separability Score')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, score in zip(bars, separability_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(separability_scores)*0.01,
                f'{score:.2f}', ha='center', va='bottom', rotation=45)
    
    plt.subplot(2, 1, 2)
    # Class distance matrix visualization
    class_distances = np.zeros((len(train_data['class'].unique()), len(train_data['class'].unique())))
    classes = train_data['class'].unique()
    
    for i, class1 in enumerate(classes):
        for j, class2 in enumerate(classes):
            if i != j:
                data1 = train_data[train_data['class'] == class1][feature_cols].mean()
                data2 = train_data[train_data['class'] == class2][feature_cols].mean()
                distance = np.linalg.norm(data1 - data2)
                class_distances[i, j] = distance
    
    sns.heatmap(class_distances, 
                xticklabels=classes, 
                yticklabels=classes,
                annot=True, 
                cmap='viridis',
                fmt='.2f')
    plt.title('Inter-Class Distance Matrix (Euclidean Distance)')
    
    plt.tight_layout()
    separability_plot_file = os.path.join(PLOTS_DIR, "17_class_separability.png")
    plt.savefig(separability_plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Class separability analysis saved to: {separability_plot_file}")
    plt.close()
    
    # Save separability results
    separability_df = pd.DataFrame({
        'feature': feature_cols,
        'separability_score': separability_scores
    }).sort_values('separability_score', ascending=False)
    
    separability_file = os.path.join(DATA_DIR, "class_separability.csv")
    separability_df.to_csv(separability_file, index=False)
    print(f"✅ Class separability results saved to: {separability_file}")
    
    print(f"Most separable features: {separability_df.head(3)['feature'].tolist()}")
    print(f"Least separable features: {separability_df.tail(3)['feature'].tolist()}")
    
    # 6. Train vs Test data comparison (Domain shift analysis)
    print("\n6. DOMAIN SHIFT ANALYSIS (TRAIN VS TEST)")
    print("-" * 50)
    
    # Statistical tests for distribution differences
    domain_shift_results = []
    
    for feature in feature_cols:
        train_vals = train_data[feature].values
        test_vals = test_data[feature].values
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.ks_2samp(train_vals, test_vals)
        
        # Mann-Whitney U test
        mw_stat, mw_p = stats.mannwhitneyu(train_vals, test_vals, alternative='two-sided')
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(train_vals) - 1) * np.var(train_vals) + 
                             (len(test_vals) - 1) * np.var(test_vals)) / 
                            (len(train_vals) + len(test_vals) - 2))
        cohens_d = (np.mean(train_vals) - np.mean(test_vals)) / pooled_std if pooled_std > 0 else 0
        
        domain_shift_results.append({
            'feature': feature,
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p,
            'mw_statistic': mw_stat,
            'mw_p_value': mw_p,
            'cohens_d': cohens_d,
            'domain_shift_risk': 'High' if ks_p < 0.01 else 'Medium' if ks_p < 0.05 else 'Low'
        })
        
        print(f"{feature}: KS={ks_stat:.3f} (p={ks_p:.2e}), Cohen's d={cohens_d:.3f} ({domain_shift_results[-1]['domain_shift_risk']} risk)")
    
    # Save domain shift analysis
    domain_shift_df = pd.DataFrame(domain_shift_results)
    domain_shift_file = os.path.join(DATA_DIR, "domain_shift_analysis.csv")
    domain_shift_df.to_csv(domain_shift_file, index=False)
    print(f"\n✅ Domain shift analysis saved to: {domain_shift_file}")
    
    high_risk_features = domain_shift_df[domain_shift_df['domain_shift_risk'] == 'High']['feature'].tolist()
    if high_risk_features:
        print(f"⚠️  High domain shift risk features: {high_risk_features}")
        print("   These features may negatively impact model performance on test data")
    
    # 7. Feature correlation impact analysis
    print("\n7. FEATURE CORRELATION IMPACT ANALYSIS")
    print("-" * 50)
    
    # Calculate correlation with target variable (using encoded labels)
    le = LabelEncoder()
    y_encoded = le.fit_transform(train_data['class'])
    
    target_correlations = []
    for feature in feature_cols:
        # Pearson correlation
        pearson_corr = np.corrcoef(train_data[feature], y_encoded)[0, 1]
        
        # Spearman correlation (rank-based)
        spearman_corr, _ = stats.spearmanr(train_data[feature], y_encoded)
        
        target_correlations.append({
            'feature': feature,
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr
        })
        
        print(f"{feature}: Pearson={pearson_corr:.3f}, Spearman={spearman_corr:.3f}")
    
    # Save correlation analysis
    correlation_df = pd.DataFrame(target_correlations)
    correlation_file = os.path.join(DATA_DIR, "target_correlation_analysis.csv")
    correlation_df.to_csv(correlation_file, index=False)
    print(f"\n✅ Target correlation analysis saved to: {correlation_file}")
    
    return {
        'statistical_results': statistical_results,
        'separability_scores': separability_scores,
        'domain_shift_results': domain_shift_results,
        'target_correlations': target_correlations,
        'quality_metrics': quality_df
    }

def comprehensive_feature_importance_analysis(models, tuned_models, X_train, y_train, feature_cols):
    """Analyze feature importance across all models"""
    print("\n" + "="*60)
    print("COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    all_importance_scores = {}
    all_models = {**models, **{f"{name} (Tuned)": info for name, info in tuned_models.items()}}
    
    # 1. Extract built-in feature importance
    print("\n1. BUILT-IN FEATURE IMPORTANCE")
    print("-" * 40)
    
    for model_name, model_info in all_models.items():
        if model_name in models:
            model = model_info['model']
        else:
            model = model_info['model']
        
        importance_scores = np.zeros(len(feature_cols))
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models (Random Forest, Gradient Boosting, Decision Tree)
            importance_scores = model.feature_importances_
            method = "Built-in"
            
        elif hasattr(model, 'coef_'):
            # Linear models (Logistic Regression, SVM with linear kernel)
            if len(model.coef_.shape) > 1:
                # Multi-class: use mean absolute coefficient
                importance_scores = np.mean(np.abs(model.coef_), axis=0)
            else:
                importance_scores = np.abs(model.coef_)
            method = "Coefficients"
            
        else:
            # Use permutation importance for other models
            print(f"  Computing permutation importance for {model_name}...")
            importance_scores = compute_feature_importances(model, X_train, y_train)
            method = "Permutation"
        
        all_importance_scores[model_name] = {
            'scores': importance_scores,
            'method': method
        }
        
        print(f"{model_name}: {method} importance calculated")
    
    # 2. Statistical feature selection methods
    print("\n2. STATISTICAL FEATURE SELECTION")
    print("-" * 40)
    
    # Mutual Information
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    all_importance_scores['Mutual Information'] = {
        'scores': mi_scores,
        'method': 'Statistical'
    }
    
    # Chi-square test (need to ensure non-negative values)
    X_train_positive = X_train - X_train.min(axis=0) + 1e-8  # Make all values positive
    chi2_scores, _ = chi2(X_train_positive, y_train)
    all_importance_scores['Chi-Square'] = {
        'scores': chi2_scores,
        'method': 'Statistical'
    }
    
    # ANOVA F-score (fix the calculation)
    f_scores = []
    unique_classes = np.unique(y_train)
    for feature_idx in range(X_train.shape[1]):
        feature_data = X_train[:, feature_idx]
        class_groups = [feature_data[y_train == cls] for cls in unique_classes]
        f_stat, _ = stats.f_oneway(*class_groups)
        f_scores.append(f_stat)
    
    f_scores = np.array(f_scores)
    all_importance_scores['ANOVA F-Score'] = {
        'scores': f_scores,
        'method': 'Statistical'
    }
    
    print("Statistical feature selection methods calculated")
    
    # 3. Create comprehensive visualization
    print("\n3. CREATING FEATURE IMPORTANCE VISUALIZATIONS")
    print("-" * 40)

    # --- Filter out invalid scores (None for DL models) ---
    valid_importance_scores = {
        method_name: score_info
        for method_name, score_info in all_importance_scores.items()
        if score_info['scores'] is not None
    }

    if not valid_importance_scores:
        print("⚠️ No valid feature importance scores to visualize.")
    else:
        # Normalize all scores to 0-1 range for comparison
        normalized_scores = {}
        for method_name, score_info in valid_importance_scores.items():
            scores = score_info['scores']
            if scores.max() > scores.min():
                normalized = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                normalized = scores
            normalized_scores[method_name] = normalized

        # --- Heatmap of all feature importance ---
        plt.figure(figsize=(16, 12))
        importance_matrix = np.array([normalized_scores[method] for method in normalized_scores.keys()])
        sns.heatmap(importance_matrix,
                    xticklabels=feature_cols,
                    yticklabels=list(normalized_scores.keys()),
                    annot=True,
                    fmt='.3f',
                    cmap='YlOrRd',
                    cbar_kws={'label': 'Normalized Importance Score'})
        plt.title('Feature Importance Across All Models and Methods', fontsize=14)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Models/Methods', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        importance_heatmap_file = os.path.join(PLOTS_DIR, "18_feature_importance_heatmap.png")
        plt.savefig(importance_heatmap_file, dpi=300, bbox_inches='tight')
        print(f"✅ Feature importance heatmap saved to: {importance_heatmap_file}")
        plt.close()

        # --- Individual model importance plots ---
        n_models = len(valid_importance_scores)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        plt.figure(figsize=(18, 6 * n_rows))
        for i, (method_name, score_info) in enumerate(valid_importance_scores.items()):
            plt.subplot(n_rows, n_cols, i + 1)
            scores = score_info['scores']
            colors = plt.cm.viridis(np.linspace(0, 1, len(feature_cols)))
            bars = plt.bar(feature_cols, scores, color=colors, alpha=0.7)
            plt.title(f'{method_name}\n({score_info["method"]} Method)', fontsize=11)
            plt.xlabel('Features')
            plt.ylabel('Importance Score')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            # Annotate top 3
            top_3_indices = np.argsort(scores)[-3:]
            for idx in top_3_indices:
                plt.text(idx, scores[idx] + scores.max()*0.01,
                        f'{scores[idx]:.3f}',
                        ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        individual_importance_file = os.path.join(PLOTS_DIR, "19_individual_feature_importance.png")
        plt.savefig(individual_importance_file, dpi=300, bbox_inches='tight')
        print(f"✅ Individual feature importance plots saved to: {individual_importance_file}")
        plt.close()

        # --- Feature importance ranking comparison ---
        plt.figure(figsize=(16, 10))
        rankings = {}
        for method_name, score_info in valid_importance_scores.items():
            scores = score_info['scores']
            ranking = stats.rankdata(-scores)  # Negative for descending order
            rankings[method_name] = ranking
        ranking_matrix = np.array([rankings[method] for method in rankings.keys()])
        avg_ranking = np.mean(ranking_matrix, axis=0)
        sorted_indices = np.argsort(avg_ranking)
        sorted_features = [feature_cols[i] for i in sorted_indices]
        sorted_rankings = ranking_matrix[:, sorted_indices]
        sns.heatmap(sorted_rankings,
                    xticklabels=sorted_features,
                    yticklabels=list(rankings.keys()),
                    annot=True,
                    fmt='.0f',
                    cmap='RdYlBu_r',
                    cbar_kws={'label': 'Ranking (1 = Most Important)'})
        plt.title('Feature Importance Rankings Across All Methods', fontsize=14)
        plt.xlabel('Features (Sorted by Average Ranking)', fontsize=12)
        plt.ylabel('Models/Methods', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        ranking_heatmap_file = os.path.join(PLOTS_DIR, "20_feature_ranking_comparison.png")
        plt.savefig(ranking_heatmap_file, dpi=300, bbox_inches='tight')
        print(f"✅ Feature ranking comparison saved to: {ranking_heatmap_file}")
        plt.close()

    
    # 6. Consensus feature importance
    print("\n4. CONSENSUS FEATURE IMPORTANCE ANALYSIS")
    print("-" * 40)

    # --- Filter valid scores only ---
    valid_importance_scores = {
        method_name: score_info
        for method_name, score_info in all_importance_scores.items()
        if score_info['scores'] is not None
    }

    if not valid_importance_scores:
        print("⚠️ No valid feature importance scores for consensus analysis.")
    else:
        # Rebuild normalized_scores and rankings using only valid models
        normalized_scores = {}
        for method_name, score_info in valid_importance_scores.items():
            scores = score_info['scores']
            if scores.max() > scores.min():
                normalized = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                normalized = scores
            normalized_scores[method_name] = normalized

        rankings = {}
        for method_name, score_info in valid_importance_scores.items():
            scores = score_info['scores']
            rankings[method_name] = stats.rankdata(-scores)

        importance_matrix = np.array([normalized_scores[m] for m in normalized_scores.keys()])
        ranking_matrix = np.array([rankings[m] for m in rankings.keys()])

        # --- Consensus Scores ---
        consensus_scores = {
            'mean_normalized': np.mean(importance_matrix, axis=0),
            'median_normalized': np.median(importance_matrix, axis=0),
            'mean_ranking': np.mean(ranking_matrix, axis=0),
            'borda_count': len(feature_cols) + 1 - np.mean(ranking_matrix, axis=0)
        }

        # --- Plot consensus ---
        plt.figure(figsize=(16, 10))
        for i, (consensus_name, scores) in enumerate(consensus_scores.items()):
            plt.subplot(2, 2, i + 1)
            sorted_indices = np.argsort(scores)[::-1]
            sorted_features = [feature_cols[j] for j in sorted_indices]
            sorted_scores = scores[sorted_indices]
            bars = plt.bar(range(len(sorted_features)), sorted_scores,
                        color=plt.cm.plasma(np.linspace(0, 1, len(sorted_features))),
                        alpha=0.7)
            plt.title(f'Consensus Feature Importance\n({consensus_name.replace("_", " ").title()})')
            plt.xlabel('Features (Ranked)')
            plt.ylabel('Consensus Score')
            plt.xticks(range(len(sorted_features)), sorted_features, rotation=45)
            plt.grid(True, alpha=0.3)
            for j in range(min(5, len(sorted_features))):
                plt.text(j, sorted_scores[j] + sorted_scores.max()*0.01,
                        f'{sorted_scores[j]:.3f}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
        plt.tight_layout()
        consensus_importance_file = os.path.join(PLOTS_DIR, "21_consensus_feature_importance.png")
        plt.savefig(consensus_importance_file, dpi=300, bbox_inches='tight')
        print(f"✅ Consensus feature importance saved to: {consensus_importance_file}")
        plt.close()

        # --- Save all importance scores ---
        importance_data = []
        for method_name, score_info in valid_importance_scores.items():
            for i, feature in enumerate(feature_cols):
                importance_data.append({
                    'method': method_name,
                    'feature': feature,
                    'raw_score': score_info['scores'][i],
                    'normalized_score': normalized_scores[method_name][i],
                    'ranking': rankings[method_name][i],
                    'method_type': score_info['method']
                })

        # Add consensus scores
        for consensus_name, scores in consensus_scores.items():
            for i, feature in enumerate(feature_cols):
                importance_data.append({
                    'method': f'Consensus_{consensus_name}',
                    'feature': feature,
                    'raw_score': scores[i],
                    'normalized_score': scores[i],
                    'ranking': stats.rankdata(-scores)[i],
                    'method_type': 'Consensus'
                })

        importance_df = pd.DataFrame(importance_data)
        importance_file = os.path.join(DATA_DIR, "comprehensive_feature_importance.csv")
        importance_df.to_csv(importance_file, index=False)
        print(f"\n✅ Comprehensive feature importance saved to: {importance_file}")
    
    # 8. Summary of top features
    print("\n5. FEATURE IMPORTANCE SUMMARY")
    print("-" * 40)
    
    # Find most consistently important features
    mean_ranking = np.mean(ranking_matrix, axis=0)
    top_features_indices = np.argsort(mean_ranking)[:5]
    top_features = [feature_cols[i] for i in top_features_indices]
    
    print("TOP 5 MOST CONSISTENTLY IMPORTANT FEATURES:")
    for i, feature_idx in enumerate(top_features_indices):
        feature = feature_cols[feature_idx]
        avg_rank = mean_ranking[feature_idx]
        print(f"{i+1}. {feature}: Average rank {avg_rank:.1f}")
        
        # Show scores across different methods
        print(f"   Scores across methods:")
        method_names = list(all_importance_scores.keys())
        for method_name in method_names[:5]:  # Show top 5 methods
            score_info = all_importance_scores[method_name]
            score = score_info['scores'][feature_idx]
            print(f"     {method_name}: {score:.4f}")
        print()
    
    return {
        'all_importance_scores': all_importance_scores,
        'normalized_scores': normalized_scores,
        'rankings': rankings,
        'consensus_scores': consensus_scores,
        'top_features': top_features
    }

def class_profile_correlation_analysis(train_data, test_data, feature_cols):
    """Analyze correlation and confusion patterns based on mean class profiles from 14 sensors"""
    print("\n" + "="*60)
    print("CLASS PROFILE CORRELATION & CONFUSION ANALYSIS")
    print("="*60)
    
    # 1. Calculate mean sensor profiles for each class
    print("\n1. CALCULATING CLASS SENSOR PROFILES")
    print("-" * 40)
    
    class_profiles = train_data.groupby('class')[feature_cols].mean()
    class_stds = train_data.groupby('class')[feature_cols].std()
    
    print("Mean sensor values by class:")
    print(class_profiles)
    print("\nStandard deviations by class:")
    print(class_stds)
    
    # Save class profiles
    profile_file = os.path.join(DATA_DIR, "detailed_class_profiles.csv")
    
    # Combine means and stds
    combined_profiles = pd.DataFrame()
    for class_name in class_profiles.index:
        for stat_type, data in [('mean', class_profiles), ('std', class_stds)]:
            for feature in feature_cols:
                combined_profiles.loc[f'{class_name}_{stat_type}', feature] = data.loc[class_name, feature]
    
    combined_profiles.to_csv(profile_file)
    print(f"\n✅ Detailed class profiles saved to: {profile_file}")
    
    # 2. Inter-class correlation analysis
    print("\n2. INTER-CLASS CORRELATION ANALYSIS")
    print("-" * 40)
    
    # Calculate correlations between class profiles
    class_correlations = class_profiles.T.corr()
    print("Correlation matrix between class sensor profiles:")
    print(class_correlations)
    
    # Visualize class profile correlations
    plt.figure(figsize=(10, 8))
    sns.heatmap(class_correlations, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Inter-Class Correlation Matrix\n(Based on Mean Sensor Profiles)', fontsize=14)
    plt.tight_layout()
    
    correlation_plot_file = os.path.join(PLOTS_DIR, "22_class_correlation_matrix.png")
    plt.savefig(correlation_plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Class correlation matrix saved to: {correlation_plot_file}")
    plt.close()
    
    # 3. Class similarity/confusion analysis
    print("\n3. CLASS SIMILARITY ANALYSIS")
    print("-" * 40)
    
    # Calculate different distance metrics between classes
    from scipy.spatial.distance import pdist, squareform
    
    distance_metrics = {
        'Euclidean': 'euclidean',
        'Manhattan': 'cityblock', 
        'Cosine': 'cosine',
        'Correlation': 'correlation'
    }
    
    similarity_matrices = {}
    
    plt.figure(figsize=(16, 12))
    
    for i, (metric_name, metric) in enumerate(distance_metrics.items()):
        # Calculate distance matrix
        distances = pdist(class_profiles.values, metric=metric)
        distance_matrix = squareform(distances)
        
        # Convert to similarity (1 - normalized distance)
        if metric == 'cosine' or metric == 'correlation':
            similarity_matrix = 1 - distance_matrix
        else:
            max_dist = distance_matrix.max()
            similarity_matrix = 1 - (distance_matrix / max_dist) if max_dist > 0 else np.ones_like(distance_matrix)
        
        similarity_matrices[metric_name] = similarity_matrix
        
        # Plot
        plt.subplot(2, 2, i + 1)
        sns.heatmap(similarity_matrix,
                    xticklabels=class_profiles.index,
                    yticklabels=class_profiles.index,
                    annot=True,
                    fmt='.3f',
                    cmap='YlOrRd',
                    square=True,
                    cbar_kws={'label': 'Similarity Score'})
        plt.title(f'{metric_name} Similarity Matrix')
        
        # Print similarity scores
        print(f"\n{metric_name} Similarity Scores:")
        classes = class_profiles.index.tolist()
        for j in range(len(classes)):
            for k in range(j+1, len(classes)):
                sim_score = similarity_matrix[j, k]
                print(f"  {classes[j]} vs {classes[k]}: {sim_score:.3f}")
    
    plt.tight_layout()
    similarity_plot_file = os.path.join(PLOTS_DIR, "23_class_similarity_matrices.png")
    plt.savefig(similarity_plot_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Class similarity matrices saved to: {similarity_plot_file}")
    plt.close()
    
    # 4. Sensor-wise class discrimination analysis
    print("\n4. SENSOR-WISE CLASS DISCRIMINATION")
    print("-" * 40)
    
    discrimination_scores = []
    
    for feature in feature_cols:
        feature_values = class_profiles[feature].values
        
        # Calculate coefficient of variation across classes
        mean_val = np.mean(feature_values)
        std_val = np.std(feature_values)
        cv = std_val / mean_val if mean_val != 0 else 0
        
        # Calculate range ratio
        range_val = np.max(feature_values) - np.min(feature_values)
        range_ratio = range_val / np.max(feature_values) if np.max(feature_values) != 0 else 0
        
        # Calculate relative standard deviation
        rel_std = std_val / np.mean(np.abs(feature_values)) if np.mean(np.abs(feature_values)) != 0 else 0
        
        discrimination_scores.append({
            'feature': feature,
            'coefficient_variation': cv,
            'range_ratio': range_ratio,
            'relative_std': rel_std,
            'mean_across_classes': mean_val,
            'std_across_classes': std_val
        })
        
        print(f"{feature}: CV={cv:.3f}, Range Ratio={range_ratio:.3f}, Rel Std={rel_std:.3f}")
    
    # Convert to DataFrame and save
    discrimination_df = pd.DataFrame(discrimination_scores)
    discrimination_file = os.path.join(DATA_DIR, "sensor_discrimination_analysis.csv")
    discrimination_df.to_csv(discrimination_file, index=False)
    print(f"\n✅ Sensor discrimination analysis saved to: {discrimination_file}")
    
    # 5. Radar chart for class profiles
    print("\n5. CREATING CLASS PROFILE RADAR CHARTS")
    print("-" * 40)
    
    # Normalize profiles for radar chart (0-1 scale)
    normalized_profiles = class_profiles.copy()
    for feature in feature_cols:
        min_val = normalized_profiles[feature].min()
        max_val = normalized_profiles[feature].max()
        if max_val > min_val:
            normalized_profiles[feature] = (normalized_profiles[feature] - min_val) / (max_val - min_val)
        else:
            normalized_profiles[feature] = 0.5  # If no variation, set to middle
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Number of features
    N = len(feature_cols)
    
    # Angle for each feature
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Colors for each class
    colors = ['red', 'blue', 'green']
    
    for i, (class_name, profile) in enumerate(normalized_profiles.iterrows()):
        values = profile.tolist()
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=class_name, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    # Add feature labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_cols)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Class Sensor Profiles (Normalized)\nRadar Chart Comparison', size=16, pad=20)
    
    radar_plot_file = os.path.join(PLOTS_DIR, "24_class_profile_radar.png")
    plt.savefig(radar_plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Class profile radar chart saved to: {radar_plot_file}")
    plt.close()
    
    # 6. Test data projection onto class profiles
    print("\n6. TEST DATA PROJECTION ANALYSIS")
    print("-" * 40)
    
    # Calculate similarity of test samples to each class profile
    test_similarities = {}
    
    for metric_name, metric in distance_metrics.items():
        similarities = []
        
        for idx, test_sample in test_data.iterrows():
            test_values = test_sample[feature_cols].values
            
            # Ensure test_values is a proper numpy array
            if not isinstance(test_values, np.ndarray):
                test_values = np.array(test_values)
            
            # Handle case where we might have a single value or malformed data
            if test_values.ndim == 0:
                test_values = np.array([test_values])
            elif test_values.ndim > 1:
                test_values = test_values.flatten()
            
            # Ensure we have the right number of features
            if len(test_values) != len(feature_cols):
                print(f"Warning: Sample {test_sample.get('sample_id', idx)} has {len(test_values)} features, expected {len(feature_cols)}")
                continue
                
            sample_similarities = {}
            
            for class_name in class_profiles.index:
                class_values = class_profiles.loc[class_name].values
                
                # Ensure class_values is also a proper array
                if not isinstance(class_values, np.ndarray):
                    class_values = np.array(class_values)
                
                if class_values.ndim == 0:
                    class_values = np.array([class_values])
                elif class_values.ndim > 1:
                    class_values = class_values.flatten()
                
                # Calculate distance/similarity
                try:
                    if metric == 'euclidean':
                        dist = np.linalg.norm(test_values - class_values)
                        max_possible_dist = np.linalg.norm(class_profiles.max().values - class_profiles.min().values)
                        similarity = 1 - (dist / max_possible_dist) if max_possible_dist > 0 else 1
                        
                    elif metric == 'cosine':
                        # Handle potential zero vectors
                        test_norm = np.linalg.norm(test_values)
                        class_norm = np.linalg.norm(class_values)
                        
                        if test_norm == 0 or class_norm == 0:
                            similarity = 0
                        else:
                            dot_product = np.dot(test_values, class_values)
                            similarity = dot_product / (test_norm * class_norm)
                            similarity = (similarity + 1) / 2  # Convert from [-1,1] to [0,1]
                            
                    elif metric == 'correlation':
                        # Fixed correlation calculation
                        try:
                            if len(test_values) > 1 and len(class_values) > 1:
                                # Ensure we have proper arrays for correlation
                                test_array = np.asarray(test_values).flatten()
                                class_array = np.asarray(class_values).flatten()
                                
                                # Check for constant arrays (zero variance)
                                if np.std(test_array) == 0 or np.std(class_array) == 0:
                                    similarity = 0
                                else:
                                    # Calculate correlation with proper array handling
                                    if len(test_array) == len(class_array):
                                        corr = np.corrcoef(test_array, class_array)[0, 1]
                                        if np.isnan(corr) or np.isinf(corr):
                                            similarity = 0
                                        else:
                                            similarity = (corr + 1) / 2  # Convert from [-1,1] to [0,1]
                                    else:
                                        similarity = 0
                            else:
                                similarity = 0
                        except Exception:
                            # Fallback for any correlation calculation errors
                            similarity = 0
                            
                    else:  # cityblock (Manhattan)
                        dist = np.sum(np.abs(test_values - class_values))
                        max_possible_dist = np.sum(np.abs(class_profiles.max().values - class_profiles.min().values))
                        similarity = 1 - (dist / max_possible_dist) if max_possible_dist > 0 else 1
                    
                    # Ensure similarity is a valid number
                    if np.isnan(similarity) or np.isinf(similarity):
                        similarity = 0
                    
                    # Clamp similarity to [0, 1] range
                    similarity = max(0, min(1, similarity))
                    
                except Exception as e:
                    print(f"Warning: Error calculating {metric} similarity for sample {test_sample.get('sample_id', idx)} and class {class_name}: {e}")
                    similarity = 0
                
                sample_similarities[class_name] = similarity
            
            similarities.append(sample_similarities)
        
        test_similarities[metric_name] = similarities
    
    # Create visualization of test sample similarities
    plt.figure(figsize=(16, 12))
    
    for i, (metric_name, similarities) in enumerate(test_similarities.items()):
        plt.subplot(2, 2, i + 1)
        
        # Create matrix: rows = test samples, columns = classes
        similarity_matrix = np.zeros((len(test_data), len(class_profiles.index)))
        
        for sample_idx, sample_sim in enumerate(similarities):
            for class_idx, class_name in enumerate(class_profiles.index):
                similarity_matrix[sample_idx, class_idx] = sample_sim[class_name]
        
        sns.heatmap(similarity_matrix,
                    xticklabels=class_profiles.index,
                    yticklabels=test_data['sample_id'].tolist(),
                    annot=True,
                    fmt='.3f',
                    cmap='YlGnBu',
                    cbar_kws={'label': 'Similarity Score'})
        plt.title(f'Test Sample Similarities ({metric_name})')
        plt.xlabel('Training Classes')
        plt.ylabel('Test Samples')
    
    plt.tight_layout()
    test_similarity_plot_file = os.path.join(PLOTS_DIR, "25_test_sample_similarities.png")
    plt.savefig(test_similarity_plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Test sample similarities saved to: {test_similarity_plot_file}")
    plt.close()
    
    # 7. Save all similarity results
    all_similarity_data = []
    
    for metric_name, similarities in test_similarities.items():
        for sample_idx, sample_sim in enumerate(similarities):
            sample_id = test_data.iloc[sample_idx]['sample_id']
            for class_name, similarity in sample_sim.items():
                all_similarity_data.append({
                    'sample_id': sample_id,
                    'metric': metric_name,
                    'class': class_name,
                    'similarity_score': similarity
                })
    
    similarity_results_df = pd.DataFrame(all_similarity_data)
    similarity_results_file = os.path.join(DATA_DIR, "test_sample_class_similarities.csv")
    similarity_results_df.to_csv(similarity_results_file, index=False)
    print(f"\n✅ Test sample class similarities saved to: {similarity_results_file}")
    
    # 8. Summary of findings
    print("\n7. CORRELATION & SIMILARITY SUMMARY")
    print("-" * 40)
    
    # Most similar classes
    euclidean_sim = similarity_matrices['Euclidean']
    np.fill_diagonal(euclidean_sim, 0)  # Remove self-similarity
    max_sim_idx = np.unravel_index(np.argmax(euclidean_sim), euclidean_sim.shape)
    classes = class_profiles.index.tolist()
    most_similar_pair = (classes[max_sim_idx[0]], classes[max_sim_idx[1]])
    max_similarity = euclidean_sim[max_sim_idx]
    
    print(f"Most similar class pair: {most_similar_pair[0]} vs {most_similar_pair[1]}")
    print(f"Euclidean similarity: {max_similarity:.3f}")
    
    # Most discriminative sensors
    discrimination_df_sorted = discrimination_df.sort_values('coefficient_variation', ascending=False)
    print(f"\nMost discriminative sensors (by coefficient of variation):")
    for i, (_, row) in enumerate(discrimination_df_sorted.head(5).iterrows()):
        print(f"{i+1}. {row['feature']}: CV = {row['coefficient_variation']:.3f}")
    
    # Average similarities for each test sample
    print(f"\nTest sample classification tendencies (Euclidean similarity):")
    euclidean_similarities = test_similarities['Euclidean']
    for sample_idx, sample_sim in enumerate(euclidean_similarities):
        sample_id = test_data.iloc[sample_idx]['sample_id']
        best_match = max(sample_sim, key=sample_sim.get)
        best_score = sample_sim[best_match]
        print(f"{sample_id}: Most similar to {best_match} (similarity: {best_score:.3f})")
    
    return {
        'class_profiles': class_profiles,
        'class_correlations': class_correlations,
        'similarity_matrices': similarity_matrices,
        'discrimination_scores': discrimination_df,
        'test_similarities': test_similarities,
        'most_similar_pair': most_similar_pair,
        'max_similarity': max_similarity
    }

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

def calculate_comprehensive_metrics(y_true, y_pred, class_labels):
    """Calculate comprehensive evaluation metrics"""
    from sklearn.metrics import (precision_score, recall_score, f1_score, 
                                matthews_corrcoef, accuracy_score, confusion_matrix)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Calculate specificity (per class, then weighted average)
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    specificities = []
    
    for i, class_label in enumerate(class_labels):
        # For multiclass: specificity = TN / (TN + FP)
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        
        if (tn + fp) > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 0.0
        specificities.append(specificity)
    
    # Weighted average specificity
    class_counts = [np.sum(y_true == label) for label in class_labels]
    total_samples = sum(class_counts)
    weighted_specificity = sum(spec * count / total_samples 
                              for spec, count in zip(specificities, class_counts))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': weighted_specificity,
        'f1_score': f1,
        'mcc': mcc
    }

def create_confusion_matrices_visualization(models, X_val, y_val, class_labels, label_encoder=None):
    """Create confusion matrix visualizations for all models (ML + Deep Learning)"""
    print("\n" + "="*50)
    print("CREATING CONFUSION MATRICES")
    print("="*50)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    model_names = list(models.keys())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for idx, (model_name, model_info) in enumerate(models.items()):
        if idx >= 6:  # Limit to 6 plots
            break

        model = model_info['model']

        # ---- Prediction step ----
        if isinstance(model, nn.Module):  # PyTorch model
            model.eval()
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            with torch.no_grad():
                outputs = model(X_val_tensor)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

            # If label encoder provided, convert back to original labels
            if label_encoder is not None:
                y_pred = label_encoder.inverse_transform(preds)
            else:
                y_pred = preds

        else:  # Scikit-learn model
            y_pred = model.predict(X_val)

        # ---- Confusion Matrix ----
        cm = confusion_matrix(y_val, y_pred, labels=class_labels)
        accuracy = accuracy_score(y_val, y_pred)

        ax = axes[idx]

        total_samples = np.sum(cm)
        annotations = []
        for i in range(len(class_labels)):
            row_annotations = []
            for j in range(len(class_labels)):
                count = cm[i, j]
                percentage = (count / total_samples) * 100
                row_annotations.append(f"{count}\n{percentage:.1f}%")
            annotations.append(row_annotations)

        colors = np.zeros_like(cm, dtype=float)
        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                colors[i, j] = 1.0 if i == j else 0.3

        im = ax.imshow(colors, cmap='RdYlGn', alpha=0.7, vmin=0, vmax=1)

        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                text_color = "white" if i == j else "black"
                weight = "bold" if i == j else "normal"
                ax.text(j, i, annotations[i][j],
                        ha="center", va="center",
                        color=text_color, fontsize=10, weight=weight)

        ax.set_xticks(range(len(class_labels)))
        ax.set_yticks(range(len(class_labels)))
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_title(f"{model_name}\nAccuracy: {accuracy:.1%}", fontsize=12, fontweight="bold")

        ax.set_xticks(np.arange(len(class_labels)+1)-0.5, minor=True)
        ax.set_yticks(np.arange(len(class_labels)+1)-0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
        ax.tick_params(which="minor", size=0)

    for idx in range(len(models), 6):
        axes[idx].set_visible(False)

    plt.tight_layout()
    confusion_matrices_file = os.path.join(PLOTS_DIR, "08_confusion_matrices.png")
    plt.savefig(confusion_matrices_file, dpi=300, bbox_inches="tight")
    print(f"✅ Confusion matrices saved to: {confusion_matrices_file}")
    plt.close()

def train_deep_model(model_class, X_train, y_train, X_val, y_val, num_classes,
                     epochs=50, batch_size=32, lr=0.001):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor),
                            batch_size=batch_size)

    model = model_class(X_train.shape[1], num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Final validation predictions
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)

    return model, np.array(all_preds)


def train_multiple_models(X_train, X_val, y_train, y_val):
    """Train multiple ML + Deep Learning models with comprehensive evaluation"""
    print("\n" + "="*50)
    print("MODEL TRAINING AND EVALUATION")
    print("="*50)

    # ---------------- Classical ML Models ----------------
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': SVC(random_state=42, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Neural Network (sklearn-MLP)': MLPClassifier(random_state=42, max_iter=1000),
        'Naive Bayes': GaussianNB()
    }

    class_labels = sorted(y_train.unique())
    results = {}
    trained_models = {}

    print(f"Training {len(models)} classical ML models: {', '.join(models.keys())}")
    print("-" * 60)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        val_pred = model.predict(X_val)
        metrics = calculate_comprehensive_metrics(y_val, val_pred, class_labels)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

        results[name] = {
            'validation_accuracy': metrics['accuracy'],
            'validation_precision': metrics['precision'],
            'validation_recall': metrics['recall'],
            'validation_specificity': metrics['specificity'],
            'validation_f1': metrics['f1_score'],
            'validation_mcc': metrics['mcc'],
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model': model
        }

        print(f"Validation Results: {metrics}")

    # ---------------- Deep Learning Models ----------------
    print("\n" + "="*50)
    print("DEEP LEARNING MODEL TRAINING")
    print("="*50)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    num_classes = len(le.classes_)

    deep_models = {
        "Deep MLP": DeepMLP,
        "Conv1D Net": Conv1DNet,
        "LSTM Net": LSTMNet
    }

    for name, model_class in deep_models.items():
        print(f"\nTraining {name}...")
        dl_model, val_preds = train_deep_model(
            model_class,
            X_train, pd.Series(y_train_enc),
            X_val, pd.Series(y_val_enc),
            num_classes=num_classes,
            epochs=50
        )

        val_pred_labels = le.inverse_transform(val_preds)
        metrics = calculate_comprehensive_metrics(y_val, val_pred_labels, le.classes_)

        results[name] = {
            'validation_accuracy': metrics['accuracy'],
            'validation_precision': metrics['precision'],
            'validation_recall': metrics['recall'],
            'validation_specificity': metrics['specificity'],
            'validation_f1': metrics['f1_score'],
            'validation_mcc': metrics['mcc'],
            'cv_mean': np.nan,  # not using CV for DL
            'cv_std': np.nan,
            'model': dl_model
        }

        trained_models[name] = dl_model
        print(f"Validation Results ({name}): {metrics}")

    # ---------------- Confusion Matrices ----------------
    create_confusion_matrices_visualization(results, X_val, y_val, class_labels, label_encoder=le)

    # ---------------- Save Performance ----------------
    performance_data = []
    for name, result in results.items():
        performance_data.append({
            'model': name,
            'validation_accuracy': result['validation_accuracy'],
            'validation_precision': result['validation_precision'],
            'validation_recall': result['validation_recall'],
            'validation_specificity': result['validation_specificity'],
            'validation_f1': result['validation_f1'],
            'validation_mcc': result['validation_mcc'],
            'cv_mean': result['cv_mean'],
            'cv_std': result['cv_std']
        })

    performance_df = pd.DataFrame(performance_data)
    performance_file = os.path.join(DATA_DIR, "model_performance.csv")
    performance_df.to_csv(performance_file, index=False)
    print(f"\n✅ Model performance results saved to: {performance_file}")

    return results, trained_models

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for selected models"""
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING")
    print("="*50)
    
    # Parameter grids for selected models
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
        'K-Nearest Neighbors': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        'Neural Network': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    }
    
    tuned_models = {}
    
    for model_name, param_grid in param_grids.items():
        print(f"\nTuning {model_name}...")
        
        if model_name == 'Random Forest':
            base_model = RandomForestClassifier(random_state=42)
        elif model_name == 'Support Vector Machine':
            base_model = SVC(random_state=42, probability=True)
        elif model_name == 'K-Nearest Neighbors':
            base_model = KNeighborsClassifier()
        elif model_name == 'Neural Network':
            base_model = MLPClassifier(random_state=42, max_iter=1000)
        
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
    
    # Note: Naive Bayes doesn't need hyperparameter tuning, so we'll add it with default parameters
    print(f"\nAdding Naive Bayes with default parameters...")
    nb_model = GaussianNB()
    nb_scores = cross_val_score(nb_model, X_train, y_train, cv=3, scoring='accuracy')
    nb_model.fit(X_train, y_train)
    
    tuned_models['Naive Bayes'] = {
        'model': nb_model,
        'best_params': 'Default parameters (no tuning needed)',
        'best_score': nb_scores.mean()
    }
    print(f"Naive Bayes CV score: {nb_scores.mean():.4f}")
    
    # Save hyperparameter tuning results
    tuning_data = []
    for name, result in tuned_models.items():
        tuning_data.append({
            'model': name,
            'best_score': result['best_score'],
            'best_params': str(result['best_params'])
        })
    
    tuning_df = pd.DataFrame(tuning_data)
    tuning_file = os.path.join(DATA_DIR, "hyperparameter_tuning.csv")
    tuning_df.to_csv(tuning_file, index=False)
    print(f"\n✅ Hyperparameter tuning results saved to: {tuning_file}")
    
    return tuned_models

def predict_unclassified_samples(models, tuned_models, X_test, test_sample_ids, label_encoder=None):
    """Make predictions for unclassified samples with ML + Deep Learning models"""
    print("\n" + "="*50)
    print("PREDICTING UNCLASSIFIED SAMPLES")
    print("="*50)

    all_predictions = {}
    prediction_probabilities = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Original Models ----------------
    for name, model_info in models.items():
        model = model_info['model']
        print(f"\n{name}:")

        # --- Scikit-learn ---
        if not isinstance(model, nn.Module):
            y_pred = model.predict(X_test)

            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
                max_probabilities = np.max(y_pred_proba, axis=1)
                avg_confidence = np.mean(max_probabilities)
                prediction_probabilities[name] = y_pred_proba
            else:
                max_probabilities, avg_confidence = None, None

        # --- PyTorch ---
        else:
            model.eval()
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            with torch.no_grad():
                outputs = model(X_test_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

            if label_encoder is not None:
                y_pred = label_encoder.inverse_transform(preds)
            else:
                y_pred = preds

            max_probabilities = np.max(probs, axis=1)
            avg_confidence = np.mean(max_probabilities)
            prediction_probabilities[name] = probs

        # Store results
        all_predictions[name] = {
            'predictions': y_pred,
            'probabilities': max_probabilities,
            'avg_confidence': avg_confidence,
            'type': 'original'
        }

        print(f"Average Prediction Confidence: {avg_confidence:.4f}" if avg_confidence else "N/A")
        for sample_id, pred, prob in zip(test_sample_ids, y_pred,
                                         max_probabilities if max_probabilities is not None else [None]*len(y_pred)):
            prob_str = f" (confidence: {prob:.3f})" if prob is not None else ""
            print(f"  {sample_id}: {pred}{prob_str}")

    # ---------------- Tuned Models ----------------
    for name, model_info in tuned_models.items():
        model = model_info['model']
        print(f"\n{name} (Tuned):")

        if not isinstance(model, nn.Module):  # sklearn tuned
            y_pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
                max_probabilities = np.max(y_pred_proba, axis=1)
                avg_confidence = np.mean(max_probabilities)
                prediction_probabilities[f"{name} (Tuned)"] = y_pred_proba
            else:
                max_probabilities, avg_confidence = None, None
        else:  # PyTorch tuned
            model.eval()
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            with torch.no_grad():
                outputs = model(X_test_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

            if label_encoder is not None:
                y_pred = label_encoder.inverse_transform(preds)
            else:
                y_pred = preds

            max_probabilities = np.max(probs, axis=1)
            avg_confidence = np.mean(max_probabilities)
            prediction_probabilities[f"{name} (Tuned)"] = probs

        all_predictions[f"{name} (Tuned)"] = {
            'predictions': y_pred,
            'probabilities': max_probabilities,
            'avg_confidence': avg_confidence,
            'type': 'tuned'
        }

        print(f"Average Prediction Confidence: {avg_confidence:.4f}" if avg_confidence else "N/A")
        for sample_id, pred, prob in zip(test_sample_ids, y_pred,
                                         max_probabilities if max_probabilities is not None else [None]*len(y_pred)):
            prob_str = f" (confidence: {prob:.3f})" if prob is not None else ""
            print(f"  {sample_id}: {pred}{prob_str}")

    # ---------------- Save to CSV ----------------
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
    predictions_file = os.path.join(DATA_DIR, "all_predictions.csv")
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
    confidence_plot_file = os.path.join(PLOTS_DIR, "09_model_confidence.png")
    plt.savefig(confidence_plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Model confidence plot saved to: {confidence_plot_file}")
    plt.close()
    
    # 2. Prediction distribution across classes
    plt.figure(figsize=(14, 8))
    all_model_predictions = {}
    for name, results in all_predictions.items():
        pred_counts = np.unique(results['predictions'], return_counts=True)
        all_model_predictions[name] = dict(zip(pred_counts[0], pred_counts[1]))
    
    training_classes = ['WFB', 'ADB', 'UFB']
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
    
    pred_dist_file = os.path.join(PLOTS_DIR, "10_prediction_distribution.png")
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
    agreement_file = os.path.join(PLOTS_DIR, "11_model_agreement.png")
    plt.savefig(agreement_file, dpi=300, bbox_inches='tight')
    print(f"✅ Model agreement plot saved to: {agreement_file}")
    plt.close()
    
    # 4. Individual sample predictions heatmap
    plt.figure(figsize=(14, 8))
    prediction_matrix = np.zeros((len(test_sample_ids), len(model_names)))
    class_to_num = {'WFB': 0, 'ADB': 1, 'UFB': 2}
    
    for j, model_name in enumerate(model_names):
        for i, pred in enumerate(all_predictions[model_name]['predictions']):
            prediction_matrix[i, j] = class_to_num.get(pred, -1)
    
    sns.heatmap(prediction_matrix, 
                xticklabels=[name.replace(' (Tuned)', '(T)') for name in model_names],
                yticklabels=test_sample_ids,
                cmap='viridis', 
                cbar_kws={'label': 'Predicted Class (0=WFB, 1=ADB, 2=UFB)'},
                annot=True, fmt='.0f')
    plt.title('Prediction Heatmap', fontsize=14)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Samples', fontsize=12)
    plt.tight_layout()
    
    heatmap_file = os.path.join(PLOTS_DIR, "12_prediction_heatmap.png")
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
        conf_dist_file = os.path.join(PLOTS_DIR, f"13_confidence_dist_{idx+1}_{model_name.replace(' ', '_')}.png")
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
    sample_conf_file = os.path.join(PLOTS_DIR, "14_per_sample_confidence.png")
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
    colors_consensus = {'WFB': 'red', 'ADB': 'blue', 'UFB': 'green'}
    
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
    consensus_file = os.path.join(PLOTS_DIR, "15_consensus_predictions.png")
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
    consensus_file = os.path.join(DATA_DIR, "consensus_predictions.csv")
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
        
        # Comprehensive data analysis
        comprehensive_analysis = comprehensive_data_analysis(train_data, test_data, feature_cols)
        
        # Prepare data for modeling
        X_train_full, X_test, y_train_full, test_sample_ids, X_train, X_val, y_train, y_val, scaler = prepare_data_for_modeling(
            train_data, test_data, feature_cols
        )
        
        # Train multiple models
        model_results, trained_models = train_multiple_models(X_train, X_val, y_train, y_val)
        
        # Hyperparameter tuning
        tuned_models = hyperparameter_tuning(X_train_full, y_train_full)
        
        # Comprehensive feature importance analysis
        feature_importance_analysis = comprehensive_feature_importance_analysis(
            model_results, tuned_models, X_train_full, y_train_full, feature_cols
        )
        
        # Class profile correlation and confusion analysis
        class_analysis = class_profile_correlation_analysis(train_data, test_data, feature_cols)
        
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
        
        # Feature importance from comprehensive analysis
        top_features = feature_importance_analysis['top_features']
        
        print(f"\nTOP 5 MOST CONSISTENTLY IMPORTANT SENSOR CHANNELS:")
        print("(Based on consensus across all models and statistical methods)")
        for i, feature in enumerate(top_features[:5]):
            feature_idx = feature_cols.index(feature)
            ranking_matrix = np.array([feature_importance_analysis['rankings'][method] for method in feature_importance_analysis['rankings'].keys()])
            avg_rank = np.mean(ranking_matrix, axis=0)[feature_idx]
            print(f"{i+1}. {feature}: Average rank {avg_rank:.1f} across all methods")
        
        print(f"\nDATASET CHARACTERISTICS:")
        print(f"- Training samples: {len(train_data)} (WFB: 40, ADB: 60, UFB: 30)")
        print(f"- Unclassified cocoa bean samples: {len(test_data)} (X1-X10)")
        print(f"- Sensor channels: {len(feature_cols)} (ch0-ch13)")
        print(f"- Training classes: {', '.join(train_data['class'].unique())}")
        print(f"- Models analyzed: Random Forest, SVM, KNN, Neural Network, Naive Bayes")
        
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
        final_results_file = os.path.join(DATA_DIR, "final_classification_results.csv")
        results_df.to_csv(final_results_file, index=False)
        print(f"\n✅ Final classification results saved to: {final_results_file}")
        
        print(f"\n" + "="*60)
        print("INTERPRETATION:")
        print("- WFB, ADB, UFB likely represent different types/qualities of cocoa beans")
        print("  * WFB: Well-Fermented Beans")
        print("  * ADB: Adequately-Fermented Beans") 
        print("  * UFB: Under-Fermented Beans")
        print("- X1-X10 are unclassified cocoa bean samples")
        print("- The model predicts which known category each sample belongs to")
        print("- Higher confidence scores indicate more reliable predictions")
        print("- Consider validating results with domain experts")
        print("="*60)
        
        # Create comprehensive summary report
        create_summary_report(
            timestamp=timestamp, 
            train_data=train_data, 
            test_data=test_data, 
            feature_cols=feature_cols,
            best_model=best_model, 
            results_df=results_df, 
            feature_importance_analysis=feature_importance_analysis, 
            comprehensive_analysis=comprehensive_analysis
        )
        
        # Close logging
        print(f"\n" + "="*60)
        print("ANALYSIS COMPLETE - ALL RESULTS SAVED")
        print("="*60)
        print(f"📁 Results directory: {RESULTS_DIR}")
        print(f"📊 Individual plots (25 files) saved in: {PLOTS_DIR}")
        print(f"📋 Data files (15 CSV files) saved in: {DATA_DIR}")
        print(f"📝 Log files saved in: {LOGS_DIR}")
        print(f"📄 Summary report: ANALYSIS_SUMMARY_REPORT_{timestamp}.txt")
        print("="*60)
        print("\n🎉 Comprehensive analysis complete with confusion matrices!")
        print("   • Focus on 5 ML models: Random Forest, SVM, KNN, Neural Network, Naive Bayes")
        print("   • Comprehensive evaluation: Accuracy, Precision, Recall, Specificity, F1-Score, MCC")
        print("   • Confusion matrices visualization similar to research papers")
        print("   • Updated class labels: WFB (Well-Fermented), ADB (Adequately-Fermented), UFB (Under-Fermented)")
        print("   • Enhanced hyperparameter tuning for better model performance")
        print("   • Individual plots covering all aspects including class correlations")
        print("   • 15 CSV files with detailed results and comprehensive metrics")
        print("   Each file can be used separately in presentations or publications.")
        
    except Exception as e:
        print(f"\n❌ Error occurred during analysis: {str(e)}")
        print("Check the log file for detailed error information.")
    
    finally:
        # Ensure logging is properly closed
        if 'logger' in globals():
            logger.close()
            sys.stdout = sys.__stdout__

def create_summary_report(timestamp, train_data, test_data, feature_cols, best_model, results_df, feature_importance_analysis, comprehensive_analysis):
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
        f.write(f"  - WFB: {len(train_data[train_data['class'] == 'WFB'])}\n")
        f.write(f"  - ADB: {len(train_data[train_data['class'] == 'ADB'])}\n")
        f.write(f"  - UFB: {len(train_data[train_data['class'] == 'UFB'])}\n")
        f.write(f"Unclassified Samples: {len(test_data)} (X1-X10)\n")
        f.write(f"Features: {len(feature_cols)} sensor channels (ch0-ch13)\n")
        f.write(f"Models Analyzed: Random Forest, SVM, KNN, Neural Network, Naive Bayes\n\n")
        
        # Data Quality Assessment
        f.write("DATA QUALITY ASSESSMENT:\n")
        f.write("-"*40 + "\n")
        quality_issues = []
        
        # Check for high domain shift risk features
        high_risk_features = [result['feature'] for result in comprehensive_analysis['domain_shift_results'] 
                             if result['domain_shift_risk'] == 'High']
        if high_risk_features:
            quality_issues.append(f"High domain shift risk: {', '.join(high_risk_features)}")
        
        # Check for low separability features
        separability_scores = comprehensive_analysis['separability_scores']
        low_sep_indices = [i for i, score in enumerate(separability_scores) if score < np.percentile(separability_scores, 25)]
        low_sep_features = [feature_cols[i] for i in low_sep_indices]
        if low_sep_features:
            quality_issues.append(f"Low class separability: {', '.join(low_sep_features[:3])}...")
        
        if quality_issues:
            for issue in quality_issues:
                f.write(f"⚠️  {issue}\n")
        else:
            f.write("✅ No major data quality issues detected\n")
        f.write("\n")
        
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
        
        # Comprehensive Feature Importance
        f.write("TOP 10 MOST CONSISTENTLY IMPORTANT FEATURES:\n")
        f.write("-"*40 + "\n")
        top_features = feature_importance_analysis['top_features'][:10]
        rankings = feature_importance_analysis['rankings']
        
        # Calculate average ranking for each feature
        ranking_matrix = np.array([rankings[method] for method in rankings.keys()])
        mean_ranking = np.mean(ranking_matrix, axis=0)
        
        for i, feature in enumerate(top_features):
            feature_idx = feature_cols.index(feature)
            avg_rank = mean_ranking[feature_idx]
            f.write(f"{i+1:2d}. {feature}: Average rank {avg_rank:.1f}\n")
        f.write("\n")
        
        # Model Agreement on Feature Importance
        f.write("FEATURE IMPORTANCE CONSENSUS:\n")
        f.write("-"*40 + "\n")
        consensus_borda = feature_importance_analysis['consensus_scores']['borda_count']
        top_consensus_indices = np.argsort(consensus_borda)[::-1][:5]
        
        f.write("Top 5 features by consensus (Borda count):\n")
        for i, idx in enumerate(top_consensus_indices):
            feature = feature_cols[idx]
            score = consensus_borda[idx]
            f.write(f"{i+1}. {feature}: {score:.2f}\n")
        f.write("\n")
        
        # Statistical Significance
        f.write("STATISTICAL SIGNIFICANCE SUMMARY:\n")
        f.write("-"*40 + "\n")
        high_sig_features = [result['feature'] for result in comprehensive_analysis['statistical_results'] 
                           if result['significance'] == 'High']
        med_sig_features = [result['feature'] for result in comprehensive_analysis['statistical_results'] 
                          if result['significance'] == 'Medium']
        
        f.write(f"Highly significant features (p<0.001): {len(high_sig_features)}\n")
        if high_sig_features:
            f.write(f"  {', '.join(high_sig_features[:5])}{'...' if len(high_sig_features) > 5 else ''}\n")
        
        f.write(f"Moderately significant features (p<0.01): {len(med_sig_features)}\n")
        if med_sig_features:
            f.write(f"  {', '.join(med_sig_features[:5])}{'...' if len(med_sig_features) > 5 else ''}\n")
        f.write("\n")
        
        # Files Generated
        f.write("FILES GENERATED:\n")
        f.write("-"*40 + "\n")
        f.write("Individual Plots (High Resolution PNG):\n")
        f.write(f"  01-07. EDA Plots: class distribution, correlations, PCA, etc.\n")
        f.write(f"  08-14. Prediction Analysis: confidence, agreement, heatmaps\n")
        f.write(f"  15. Feature Distributions by Class\n")
        f.write(f"  16. Outlier Analysis\n")
        f.write(f"  17. Class Separability Analysis\n")
        f.write(f"  18. Feature Importance Heatmap (All Models)\n")
        f.write(f"  19. Individual Feature Importance (All Models)\n")
        f.write(f"  20. Feature Ranking Comparison\n")
        f.write(f"  21. Consensus Feature Importance\n")
        f.write(f"  22. Class Correlation Matrix\n")
        f.write(f"  23. Class Similarity Matrices\n")
        f.write(f"  24. Class Profile Radar Chart\n")
        f.write(f"  25. Test Sample Similarities\n\n")
        
        f.write("Data Files (CSV):\n")
        f.write(f"  - final_classification_results.csv\n")
        f.write(f"  - comprehensive_feature_importance.csv\n")
        f.write(f"  - detailed_statistical_analysis.csv\n")
        f.write(f"  - class_separability.csv\n")
        f.write(f"  - domain_shift_analysis.csv\n")
        f.write(f"  - data_quality_metrics.csv\n")
        f.write(f"  - target_correlation_analysis.csv\n")
        f.write(f"  - all_predictions.csv\n")
        f.write(f"  - consensus_predictions.csv\n")
        f.write(f"  - model_performance.csv\n")
        f.write(f"  - hyperparameter_tuning.csv\n")
        f.write(f"  - detailed_class_profiles.csv\n")
        f.write(f"  - sensor_discrimination_analysis.csv\n")
        f.write(f"  - test_sample_class_similarities.csv\n\n")
        
        f.write("Log Files:\n")
        f.write(f"  - analysis_log.txt\n\n")
        
        # Key Insights and Recommendations
        f.write("KEY INSIGHTS & RECOMMENDATIONS:\n")
        f.write("-"*40 + "\n")
        f.write("1. DATA QUALITY:\n")
        if not high_risk_features:
            f.write("   ✅ No significant domain shift detected between train/test\n")
        else:
            f.write(f"   ⚠️  Domain shift risk in: {', '.join(high_risk_features[:3])}\n")
            f.write("   → Consider feature scaling or domain adaptation techniques\n")
        
        f.write("\n2. FEATURE IMPORTANCE:\n")
        f.write(f"   🎯 Most critical sensors: {', '.join(top_features[:3])}\n")
        f.write("   → Focus on these sensors for future data collection\n")
        f.write("   → Consider sensor maintenance and calibration priorities\n")
        
        f.write("\n3. MODEL PERFORMANCE:\n")
        avg_confidence = best_model[1].get('avg_confidence', 0)
        if avg_confidence and avg_confidence > 0.8:
            f.write("   ✅ High model confidence - predictions are reliable\n")
        elif avg_confidence and avg_confidence > 0.6:
            f.write("   ⚠️  Moderate confidence - review uncertain predictions\n")
        else:
            f.write("   ❌ Low confidence - consider additional data or features\n")
        
        f.write("\n4. CLASSIFICATION RESULTS:\n")
        f.write("   📊 Sample distribution balanced across predicted classes\n")
        f.write("   → Validate results with domain experts\n")
        f.write("   → Consider chemical analysis verification for uncertain samples\n")
        
        f.write("\n5. NEXT STEPS:\n")
        f.write("   • Validate predictions with chemical/sensory analysis\n")
        f.write("   • Collect more data if confidence is low\n")
        f.write("   • Focus on top-ranked features for sensor optimization\n")
        f.write("   • Monitor domain shift in future data\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
        f.write("END OF COMPREHENSIVE REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"\n✅ Comprehensive summary report saved to: {report_file}")

if __name__ == "__main__":
    main()