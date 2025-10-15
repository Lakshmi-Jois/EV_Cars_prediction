 EV_Cars_prediction
Analysis of 103 Electric Vehicles with 14 features, including range, speed, and efficiency, to build a regression model for accurate EV price prediction.

 ⚡ Electric Vehicle (EV) Price Predictor

This project develops a regression model to predict the price of Electric Vehicles (EVs) in Euro (€) based on key performance and technical specifications. The analysis uses a dataset of EV specifications, focusing on robust data preprocessing and feature selection to build a stable and interpretable Linear Regression model.

---

 💾 Dataset Overview

The dataset initially contained 103 records and 14 features, capturing comprehensive details about electric cars. The target variable for prediction is PriceEuro.

| Feature | Data Type | Description |
| :--- | :--- | :--- |
| Brand | `object` | The manufacturer of the EV. |
| AccelSec | `float64` | Time to accelerate from 0-100 km/h (seconds). |
| Range_Km | `int64` | Official driving range on a single charge (km). |
| Efficiency_WhKm | `int64` | Energy efficiency (Wh/km). |
| FastCharge_KmH | `object` | Fast charging speed (km of range added per hour). |
| RapidCharge | `object` | Support for rapid charging (Yes/No). |
| PowerTrain | `object` | Drivetrain configuration (e.g., RWD, AWD). |
| PlugType | `object` | Type of charging plug. |
| BodyStyle | `object` | Vehicle body type (e.g., Sedan, Hatchback). |
| Segment | `object` | Market classification segment. |
| PriceEuro | `int64` | Target Variable: Retail price in Euros (€). |

---

 🧠 Methodology and Feature Engineering

 1. Feature Selection (Dimensionality Reduction)

During the feature selection process, three columns were deliberately dropped from the dataset to enhance model stability, efficiency, and interpretability:

| Dropped Feature | Rationale | Impact on Model |

| Model | High-cardinality categorical identifier, risking the introduction of noise and model instability. | Improves generalization and reduces data sparsity from one-hot encoding. |
| Seats | Showed very low variance and an insignificant correlation with the target variable. | Improves model efficiency by removing an uninformative feature. |
| TopSpeed_KmH | Exhibited strong multicollinearity (high VIF) with other numerical predictors (e.g., `AccelSec`). | Ensures reliable coefficient estimates and reduces redundancy in the regression analysis. |

 2. Preprocessing & Outlier Handling

* Categorical Encoding: Nominal categorical features (`Brand`, `PowerTrain`, etc.) were processed using appropriate encoding techniques.
* Outlier Management: The Robust Scaler was applied to the numerical features. This scaling technique is less sensitive to outliers than standard scaling, ensuring the model's stability against extreme data points.
* Final Model: A Linear Regression model was chosen for its simplicity and interpretability, allowing for direct insights into how key technical specs (like `Range_Km` and `AccelSec`) influence the final price.

---

 🎯 Results and Conclusion

The resulting model is built on a streamlined set of independent variables, offering clear and reliable insights into the factors that drive EV pricing. The removal of multicollinear and low-variance features has successfully led to a more stable and interpretable regression model.

---

 🚀 Getting Started

1.  Clone this repository: `git clone [Your-Repo-Link]`
2.  Install the required dependencies: `pip install pandas numpy scikit-learn`
3.  Run the main analysis script (`your_script_name.py` or Jupyter Notebook) to reproduce the model training and prediction results.
