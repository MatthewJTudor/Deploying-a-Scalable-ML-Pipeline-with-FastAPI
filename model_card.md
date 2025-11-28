# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model is a **RandomForestClassifier** trained on UCI's Adult Census Income dataset.
The implementation follows the ML pipeline defined in the project structure:
- **Encoder:** `sklearn` OneHotEncoder with `handle_unknown='ignore'`.
- **Label Processer:** `sklearn` LabelBinarizer.
- **Processing Function:** `process_data` provided in `ml/data.py`.
- **Training Script**: `train_model.py`.
- **Serialization**: Models saved using Python `pickle` in `model/model.pkl` and `model/encoder.pkl`.
## Intended Use

The model is intended exclusively for educational purposes within the context of the WGU/Udacity ML DevOps course.  
It demonstrates:

- End-to-end ML pipeline construction
    
- CI/CD integration
    
- Model metric tracking
    
- Data slicing analysis
    
- Serialization for deployment through FastAPI
    

It is not intended for deployment in any production or decision-making system.
## Training Data

The model was trained using the **Adult Census Income dataset** (“census.csv”), which originates from the **UCI Machine Learning Repository**.  
This dataset includes demographic and employment information such as:

- Age
    
- Workclass
    
- Education
    
- Marital status
    
- Occupation
    
- Relationship
    
- Race
    
- Sex
    
- Native country
    
- Hours-per-week
    
- Capital gain/loss
    

Categorical variables were encoded via one-hot encoding; continuous features were used as-is.

The dataset was split using an **80/20 train-test split** with `random_state=42`.
## Evaluation Data

Evaluation was performed on the held-out 20% test set, processed with the same encoder and label binarizer used during training.

Slice performance metrics were computed for each unique value of every categorical feature and stored in `slice_output.txt` as required.
## Metrics

The following metrics were used to evaluate the model:

- **Precision**
    
- **Recall**
    
- **F1 score**

### Actual output:

```
Precision: 0.7419
Recall:    0.6384
F1 Score:  0.6863
```

Slice-based metrics were also computed for all categorical subgroups. These results are included in `slice_output.txt`.
## Ethical Considerations

This model is intended only for instructional purposes and should not be used to make real decisions.

The dataset contains sensitive attributes such as race, sex, and native country.  
Models trained on this dataset may exhibit biases due to:

- Historical societal inequities reflected in the data
    
- Imbalanced representation across demographic groups
    
- Correlation between protected attributes and income levels
## Caveats and Recommendations

- The Adult Census dataset is over 25 years old and does not represent current socioeconomic conditions.
    
- One-hot encoding can cause sparsity and high-dimensional feature spaces; alternative encoders may improve performance.
    
- Hyperparameter tuning could increase model accuracy.