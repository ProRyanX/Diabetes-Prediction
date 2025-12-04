# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ~~~~~~~~~~Sklearn libraries for model building and evaluation~~~~~~~~~~
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

# ~~~~~~~~~~~~~~Importing various machine learning models~~~~~~~~~~~~~~
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Loading dataset and understanding its structure
data = pd.read_csv('dataset.csv')
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

# Handling missing/Zero values
cols = list(data.columns)
for col in cols:
    if data[col].isnull().sum() > 0:
        if data[col].dtype == 'object':
            data[col].fillna(data[col].mode()[0], inplace=True)
        else:
            data[col].fillna(data[col].median(), inplace=True)

# Outlier detection and treatment
numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
for col in numerical_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
    data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])

# Correlation analysis
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Encoding categorical variables
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
LE = LabelEncoder()
for col in categorical_cols:
    data[col] = LE.fit_transform(data[col])
    print(f'Encoded {col} with classes: {LE.classes_}')

# Modeling preparation
target_variable = "Outcome"
x = data.drop(columns=[target_variable])
y = data[target_variable]

# Scaling features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Splitting dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)


# ~~~~~~~~~~Model building and evaluation~~~~~~~~~
# 1.) Logistic Regression
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(x_train, y_train)
y_pred_lr = lr_model.predict(x_test)

# 2.) Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(x_train, y_train)
y_pred_dt = dt_model.predict(x_test)

# 3.) Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)

# 4.) Multi-layer Perceptron Classifier (MLP for Neural Network)
mlp_model = MLPClassifier(hidden_layer_sizes=(16,8), activation='relu', max_iter=400)
mlp_model.fit(x_train, y_train)
y_pred_mlp = mlp_model.predict(x_test)

print("\n\nModel Evaluation Results:\n")

# Function to evaluate model performance
def evaluate_model(model_name, y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print(f"====={model_name}=====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("\n")

# Evaluating all models
models = {
    "Logistic Regression": y_pred_lr,
    "Decision Tree": y_pred_dt,
    "Random Forest": y_pred_rf,
    "MLP Classifier": y_pred_mlp
}
for model_name, y_pred in models.items():
    evaluate_model(model_name, y_test, y_pred)


# ============= Visualization =============
# Displaying Confusion matrix for all models
models = {
    "Logistic Regression": (lr_model, y_pred_lr),
    "Decision Tree": (dt_model, y_pred_dt),
    "Random Forest": (rf_model, y_pred_rf),
    "MLP Classifier": (mlp_model, y_pred_mlp)
}
for model_name, (model, pred) in models.items():
    disp = ConfusionMatrixDisplay.from_predictions(y_test, pred)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

# ROC Curve for all models
plt.figure(figsize=(8,6))
plt.close('all')   # Close previous plots to avoid overlap
RocCurveDisplay.from_estimator(lr_model, x_test, y_test, name='Logistic Regression')
RocCurveDisplay.from_estimator(dt_model, x_test, y_test, name='Decision Tree')
RocCurveDisplay.from_estimator(rf_model, x_test, y_test, name='Random Forest')
RocCurveDisplay.from_estimator(mlp_model, x_test, y_test, name='MLP Classifier')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()

# Featuring importance from Random Forest
feature_importances = rf_model.feature_importances_
features = x.columns
plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importances, y=features, width=0.6)
plt.title('Feature Importances from Random Forest')
plt.show()

# Neural Network Loss Curve
plt.plot(mlp_model.loss_curve_)
plt.title('MLP Classifier Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()