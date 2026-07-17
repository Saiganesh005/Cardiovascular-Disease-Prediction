import os

# Define the content of the markdown report
report_content = """
## Output Report: Cardiovascular Disease Prediction

### 1. Introduction
This report summarizes the process of building and evaluating various machine learning models for cardiovascular disease prediction using the provided dataset. The goal was to train multiple models and identify the best-performing one.

### 2. Data Exploration and Preprocessing
- **Histograms:** Distributions of all features were visualized to understand their spread and identify potential outliers.

    <img src="output/images/histogram.png" alt="Histograms of Features" style="width: 700px;">

- **Target Distribution:** The distribution of the target variable ('cardio') showed a balanced dataset.

    <img src="output/images/countplot.png" alt="Cardio Distribution" style="width: 400px;">

- **Feature Boxplot:** A boxplot was generated to visualize the spread and outliers in the features.

    <img src="output/images/boxplot.png" alt="Feature Boxplot" style="width: 700px;">

- **Correlation Heatmap:** The correlation between all features was visualized to understand relationships and potential multicollinearity.

    <img src="output/images/correlation_heatmap.png" alt="Correlation Heatmap" style="width: 600px;">

- **Pairplot:** A pairplot for key numerical features was generated to observe relationships segmented by the 'cardio' outcome.

    <img src="output/images/pairplot.png" alt="Pairplot" style="width: 700px;">

### 3. Model Training and Evaluation
Several machine learning models, including Logistic Regression, KNN, Decision Tree, Random Forest, SVM, and an Artificial Neural Network (ANN), were trained and evaluated. The dataset was split into training and testing sets, and features were scaled using `StandardScaler`.

#### Model Performance Summary:

-   **Logistic Regression:** Accuracy: 0.7234
-   **KNN:** Accuracy: 0.6279
-   **Decision Tree:** Accuracy: 0.6362
-   **Random Forest:** Accuracy: 0.7258
-   **SVM:** Accuracy: 0.7299
-   **ANN:** Accuracy: 0.7379

**Overall Best Performing Model: ANN** with an accuracy of **0.7379**.

#### ANN Training Visualizations:

-   **Training vs Validation Accuracy:** This plot shows how the ANN's accuracy on the training and validation sets changed over epochs.

    <img src="output/images/training_accuracy.png" alt="Training vs Validation Accuracy" style="width: 500px;">

-   **Training vs Validation Loss:** This plot illustrates the training and validation loss over epochs.

    <img src="output/images/training_loss.png" alt="Training vs Validation Loss" style="width: 500px;">

#### Model Comparison:

-   **Accuracy Comparison:** A bar chart comparing the accuracies of all trained models.

    <img src="output/images/model_comparison.png" alt="Model Comparison" style="width: 500px;">

#### Other Evaluation Metrics:

-   **Confusion Matrix (ANN):** Visual representation of the ANN's classification performance on the test set.

    <img src="output/images/confusion_matrix.png" alt="Confusion Matrix" style="width: 400px;">

-   **ROC Curve (ANN):** Receiver Operating Characteristic curve for the ANN, showing its ability to distinguish between classes (AUC = 0.800).

    <img src="output/images/roc_curve.png" alt="ROC Curve" style="width: 400px;">

### 4. Conclusion
Based on the evaluation, the Artificial Neural Network (ANN) model achieved the highest accuracy of 0.7379 on the test set. All relevant models and visualizations have been saved to the `output/models` and `output/images` directories, respectively. Log outputs detailing the training process and results are available in `output/logs/model_training.log`.
"""

# Define the path for the reports directory and the report file
reports_dir = "output/reports"
report_file_path = os.path.join(reports_dir, "report.md")

# Create the reports directory if it doesn't exist
os.makedirs(reports_dir, exist_ok=True)

# Write the report content to the file
with open(report_file_path, "w") as f:
    f.write(report_content)

print(f"Output report saved to: {report_file_path}")
