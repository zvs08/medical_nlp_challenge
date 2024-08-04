# Analysis

## Statement of the Problem

The task was to develop a method to determine whether each clinical note indicates that the patient has diabetes and/or cancer. The notes are derived from the Asclepius Synthetic Clinical Notes dataset, and only some rows have labels for diabetes and cancer. The goal was to provide accurate predictions that can be used in downstream workflows.

## Assumptions and Modifications

- We assumed that the clinical notes contain enough information to make predictions about diabetes and cancer.
- For diabetes, we predicted whether the patient is currently diabetic (diabetes mellitus).
- For cancer, we predicted whether the patient has had cancer (including historically).
- We used BERT (specifically the `UFNLP/gatortronS` model) for text encoding and KNN for final predictions.

## Methodology

### Data Loading and Preprocessing

1. **Loading Data**: The data was loaded from a CSV file.
2. **Filtering**: Rows with non-null values for `has_cancer` and `has_diabetes` and where `test_set` is 0 were filtered.
3. **Splitting**: The data was split into training and validation sets proportionally to the target variables.

### Model Training

Given the constraints with the labeled data:

1. **Limited Labeled Data**: Only 50 examples of labeled data were available, which significantly impacted the model training process.
2. **BERT Models**: Two separate BERT models were trained for cancer and diabetes prediction, using a small number of steps due to the limited data.
3. **Training Arguments**: Adjusted for the small dataset size, including increased epochs and gradient accumulation steps, to make the most out of the limited data.

### Prediction and KNN

1. **Pooler Layer Extraction**: The pooler layer from the trained BERT models, which provides a 1024-dimensional vector for each input, was used for further processing.
2. **KNN Training**: KNN models were trained on the pooler layer vectors of the 50 labeled examples.
3. **KNN Prediction**: The KNN models predicted the labels for the unlabeled data based on the extracted pooler layer vectors.

## Quantitative Error Analysis

- Due to the nature of the task and the limited labeled data, cross-validation and detailed performance metrics were challenging to implement within the time constraints.
- The performance was evaluated based on the accuracy of predictions on the validation set.

## Qualitative Error Analysis

- **Ambiguity**: Some clinical notes may have ambiguous language that makes it difficult to determine the presence of diabetes or cancer.
- **Imbalanced Data**: The partially labeled data may lead to imbalanced training, affecting model performance.
- **Overfitting**: With a small dataset, there is a risk of overfitting, especially with complex models like BERT.

## Proposed Areas for Improvement and Next Steps

1. **Data Augmentation**: Increase the dataset size with more labeled examples or synthetic data to improve model training and evaluation.
2. **Model Tuning**: Experiment with different models and hyperparameters to improve performance, given more data.
3. **Cross-Validation**: Implement cross-validation to better understand the model's performance and ensure robustness.
4. **Error Analysis**: Conduct a detailed error analysis to identify specific failure modes and address them.

## Conclusion

This project demonstrated a method to predict diabetes and cancer from clinical notes using BERT and KNN. Despite the constraints posed by the limited labeled data, the approach was effective given the circumstances and provided a foundation for further improvements.
