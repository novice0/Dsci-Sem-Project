# Emissions and Agriculture Machine Learning Projects

This repository contains two machine learning workflows developed for the **Emissions** and **Agriculture** datasets. Both projects follow standard ML pipelines from preprocessing and visualization to model training and performance optimization using advanced techniques.

---

## Contents

- [`/emissions`](#emissions-project)
- [`/agriculture`](#agriculture-project)

---

## Emissions Project

### Objective:

To predict emissions using a variety of machine learning models, ranging from linear regression to deep learning.

### Steps Followed:

1. **Preprocessing**

   - Handled missing values, scaled numerical attributes, and encoded categorical variables where required.

2. **Visualization**

   - Plotted correlation matrices, scatter plots, and pair plots to analyze relationships between variables.

3. **Linear Regression**

   - Implemented as a baseline model for emissions prediction.

4. **Random Forest Regressor**

   - Improved model performance by training an ensemble model.

5. **Recurrent Neural Network (RNN)**
   - Further reduced loss and improved prediction accuracy by designing an RNN-based architecture.
   - **Layers Used:** Input layer → LSTM/GRU layers → Dense layers → Output layer
   - Read about architecture design and rationale in the source notebook.

---

## Agriculture Project

### Objective:

To classify agricultural data using ensemble learning methods and gradient boosting models.

### Steps Followed:

1. **Preprocessing**

   - Cleaned and transformed data for training and testing.

2. **Visualization**

   - Performed EDA to uncover insights and patterns in the dataset.

3. **Random Forest Classifier**

   - Used for initial classification and performance benchmarking.

4. **Gradient Boosting Models**
   - Implemented both **XGBoost** and **LightGBM** for more accurate and efficient classification.

---

## Installation & Requirements

To run the notebooks, install the required packages:

```bash
pip install -r requirements.txt
```

---

## Contact Us

For queries or collaboration ideas, feel free to open an issue or reach out via email.
