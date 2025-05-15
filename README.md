# Machine Failure Prediction using Sensor Data

This project applies machine learning models to predict whether a machine will fail based on real-time sensor inputs. The primary goal is to use **Support Vector Machines (SVM)** and **Random Forest** to model failure conditions and help with proactive maintenance.

## Dataset

**Source**: https://www.kaggle.com/datasets/umerrtx/machine-failure-prediction-using-sensor-data  
**Records**: 944 observations  
**Features**: 9 sensor-based measurements + 1 binary target variable (`fail`)

## 📊 Feature Descriptions

| Feature         | Description |
|----------------|-------------|
| **footfall**     | Number of people or movements detected. |
| **tempMode**     | Operating temperature mode setting of the machine. |
| **AQ**           | Air quality sensor reading. |
| **USS**          | Ultrasonic sensor signal. |
| **CS**           | Current sensor reading. |
| **VOC**          | Volatile Organic Compound concentration. |
| **RP**           | Rotational pressure or power level. |
| **IP**           | Input power or internal pressure. |
| **Temperature**  | Measured temperature in the system. |
| **fail**         | Target variable: 1 = failure, 0 = no failure. |

## 🔧 Data Preprocessing

- Normalized all sensor features using standard scaling.
- Split the dataset into training (70%) and testing (30%) sets.
- Handled class imbalance analysis via exploratory plots.

## 📈 Exploratory Data Analysis

### Class Distribution

- Bar chart shows a slightly imbalanced dataset with more no-failure cases.

![machine_failure_class_balance](https://github.com/user-attachments/assets/3d05b97c-4105-4e5f-b01d-d868a686029b)

### Correlation Heatmap

- Strong relationships observed between `AQ`, `VOC`, and `fail`.

![machine_failure_heatmap](https://github.com/user-attachments/assets/81501847-7abd-4be7-8f8b-624e1b7bdaed)


## 🤖 Models Used

### 1. **Support Vector Classifier (RBF Kernel)**
- Tuned using grid search on `C` and `gamma`.
- Excellent generalization and precision.

### 2. **Random Forest Classifier**
- Tuned over `max_depth` via grid search.
- Performed competitively with a slight overfitting tendency.

## Model 1: SVC with Radial Basis Function

### Confusion Matrices

![machine_failure_CM_SVC](https://github.com/user-attachments/assets/f2fb1fb4-6420-401b-b423-9b7edc9fc26a)

### Classification Report

![image](https://github.com/user-attachments/assets/c976764e-66ea-4d1b-95a2-91a490d668fd)

## Model 2: Random Forest

### Confusion Matrix

![machine_failure_CM_RF](https://github.com/user-attachments/assets/ff87180b-05ad-401f-9498-4738467e05bc)

### Classification Report

![image](https://github.com/user-attachments/assets/0dda47ec-68db-4766-af22-221f9ce2935d)


## Model Performance

![image](https://github.com/user-attachments/assets/22f0e655-5ef7-47cf-b9d4-5d60cf8b7fb1)


## Result and Interpretations

- **SVC with RBF** kernel performed slightly better than Random Forest across all metrics.
- It had higher **precision** and **F1 score**, making it more reliable for high-risk environments where false alarms are costly.
- Confusion matrices show both models are well-balanced, but SVC has fewer total misclassifications.

## Conclusion

SVC (RBF kernel) is selected as the final model due to its superior accuracy, better balance of precision and recall, and slightly fewer false predictions.  
It is well-suited for predictive maintenance systems where early detection of failure is critical.
