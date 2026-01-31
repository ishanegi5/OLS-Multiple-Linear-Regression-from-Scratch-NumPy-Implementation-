# 📘 OLS Multiple Linear Regression from Scratch (NumPy Only)

This repository contains a complete, from-scratch implementation of **Multiple Linear Regression** using the **Ordinary Least Squares (OLS)** method.  
The goal is to understand how Linear Regression works internally — without using libraries like scikit-learn.

We train and test the model on the **Diabetes dataset** (442 samples, 10 features), and implement all mathematical operations manually using NumPy.

---

## 🚀 Project Highlights

- ✔ Implement Linear Regression **without scikit-learn**  
- ✔ Use **Normal Equation** for solving coefficients  
- ✔ Custom class `MeraLR`  
- ✔ Add intercept term manually  
- ✔ Predict using pure matrix multiplication  
- ✔ Train on the **Diabetes dataset**  
- ✔ Compare results with scikit-learn  
- ✔ Beginner-friendly & fully explained

---

## 📂 Code Structure

OLS_multiple_linear_regression_from_scratch.ipynb

Inside the notebook, you will find:

- Loading the Diabetes dataset  
- Understanding feature shapes  
- Creating a custom Linear Regression class  
- Computing coefficients using:  
  \[
  \beta = (X^TX)^{-1}X^Ty
  \]
- Storing weights & intercept  
- Making predictions  
- Evaluating accuracy (RMSE, R²)  
- Comparison with scikit-learn’s `LinearRegression`

---

## 🧠 Mathematical Formula (Normal Equation)

For a given dataset:

\[
\hat{y} = X\beta
\]

The optimal weights are obtained using:

\[
\beta = (X^TX)^{-1}X^Ty
\]

Where:

- \(X\) = feature matrix  
- \(y\) = target vector  
- \(\beta\) = coefficients + intercept  

We calculate this manually using **NumPy linear algebra (`np.linalg`)**.

---

## 🛠 Implementation (Core Logic)

### Adding Intercept Column
```python
X_train = np.insert(X_train, 0, 1, axis=1)
```
Computing Coefficients
betas = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

Storing Parameters
self.intercept_ = betas[0]
self.coef_ = betas[1:]

Predicting
y_pred = np.dot(X_test, self.coef_) + self.intercept_

📊 Dataset Used — Diabetes Dataset

Total samples: 442

Total features: 10

All features are normalized

Regression target: disease progression measure

Loaded using:

from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True)

📈 Results

Model successfully computes coefficients

Predictions match scikit-learn’s Linear Regression (minor floating point differences)

Helps understand the math behind ML algorithms

🔧 Requirements
numpy
scikit-learn
matplotlib  (optional)

🧪 How to Run

Clone the repository:

git clone <your-repo-url>


Open the notebook:

OLS_multiple_linear_regression_from_scratch.ipynb


Run all cells to see training, predictions, and evaluations.

🤝 Contributing

Feel free to contribute improvements, suggestions, or optimizations!

⭐ Show Support

If this helped you understand Linear Regression better, consider giving the repository a ⭐ on GitHub!

📬 Contact

For questions or suggestions:
Isha Negi (Author)
