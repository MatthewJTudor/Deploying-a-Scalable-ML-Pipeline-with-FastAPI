# Model Card

For additional information, see the [Model Card paper.](https://arxiv.org/pdf/1810.03993.pdf)

## Model Details

The model is a random forest classifier trained on a one-hot encoded categorical feature set. All training functions used were from the sklearn library.

Models were saved using Python pickle and are stored in the model folder.

The model predicts whether an individual's income exceeds $50,000 based on demographic and socioeconomic variables.

Continuous integration is managed through GitHub workflows, integrating flake8 and pytest.

API is used to verify system status and get predictions based on specific feature values.

## Intended Use

This model is to be used solely for educational purposes.

## Training Data

The dataset, titled 'Census Income, ' is sourced from the University of California, Irvine's Machine Learning Repository. [LINK HERE](https://archive.ics.uci.edu/dataset/20/census+income)

All categorical variables were one-hot encoded. Continuous data was not considered.

The data was split into training and testing sets using an 80/20 split.

## Evaluation Data

The test set was evaluated using the same one-hot encoding and binarizing as the training set.

Slice performance metrics were also calculated over categorical features. This data is stored in slice_output.txt.

## Metrics

The metrics considered were precision, recall, and F1 score to measure predictive performance.

The model scored 0.74, 0.64, and 0.69, respectively.

Precision is somewhat intense, but recall falls behind. The 0.69 at F1 indicates a balanced yet not optimized model.

## Ethical Considerations

The dataset has not been evaluated for fairness or bias.

This model is intended for educational and evaluation purposes only.

## Caveats and Recommendations

The dataset is from 1994 and is not normalized for the current socioeconomic reality.

Sweeping the parameters of the random forest function could tune configurations which improve performance.

One-hot encoding can create relative sparsity within categories. Other encoding methods could yield better results.