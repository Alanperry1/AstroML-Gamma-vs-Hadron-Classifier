# AstroML-Gamma-vs-Hadron-Classifier

AstroML is a machine learning project that classifies cosmic particle events from the MAGIC Gamma Telescope into gamma rays or hadrons. It uses supervised algorithms, KNN, Naïve Bayes, Logistic Regression, and SVM to analyze and compare classical model performance on telescope-based astrophysical data.

## Key Features & Benefits

*   **Cosmic Particle Classification:** Classifies cosmic events into gamma rays or hadrons.
*   **Classical Machine Learning Algorithms:** Implements KNN, Naïve Bayes, Logistic Regression, and SVM.
*   **Performance Analysis:** Compares the performance of different machine learning models on astrophysical data.
*   **Telescope Data Analysis:** Analyzes data obtained from the MAGIC Gamma Telescope.

## Prerequisites & Dependencies

Before running this project, ensure you have the following installed:

*   **Python:** (Recommended version 3.7 or higher)
*   **Jupyter Notebook:** For running the included notebook.
*   **Required Python Libraries:**
    *   NumPy
    *   Pandas
    *   Scikit-learn (sklearn)
    *   Matplotlib
    *   AstroML

You can install these dependencies using pip:

```bash
pip install numpy pandas scikit-learn matplotlib astroML
```

## Installation & Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Alanperry1/AstroML-Gamma-vs-Hadron-Classifier.git
    cd AstroML-Gamma-vs-Hadron-Classifier
    ```

2.  **Install dependencies:**

    ```bash
    pip install numpy pandas scikit-learn matplotlib astroML
    ```

3.  **Run the Jupyter Notebook:**

    ```bash
    jupyter notebook Telescope.ipynb
    ```

## Usage Examples

The primary usage is through the `Telescope.ipynb` Jupyter Notebook. The notebook walks through data loading, preprocessing, model training, and evaluation.

Example snippet (from hypothetical model training):

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Assuming 'data' is a pandas DataFrame containing the features and labels
X = data.drop('label', axis=1)  # Features
y = data['label']               # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")
```

## Configuration Options

Currently, there are no specific configuration options exposed outside of the notebook parameters. You can modify parameters within the `Telescope.ipynb` notebook, such as:

*   **Model parameters:** Adjust parameters specific to each algorithm (e.g., the `n_neighbors` parameter in KNN).
*   **Data splitting ratios:** Modify the `test_size` parameter in `train_test_split`.
*   **Random seeds:** Control the randomness for reproducibility.

## Contributing Guidelines

Contributions are welcome! To contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Implement your changes.
4.  Ensure your code is well-documented and tested.
5.  Submit a pull request with a clear description of your changes.

## License Information

This project has no license specified. All rights are reserved by the owner.
## Acknowledgments

This project utilizes the MAGIC Gamma Telescope data and benefits from the AstroML library. We acknowledge the contributions of the AstroML developers and the MAGIC collaboration.
