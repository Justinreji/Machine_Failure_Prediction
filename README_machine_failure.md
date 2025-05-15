
# 🛠️ Machine Failure Prediction using Sensor Data

This project applies machine learning models to predict whether a machine will fail based on real-time sensor inputs. The primary goal is to use **Support Vector Machines (SVM)** and **Random Forest** to model failure conditions and help with proactive maintenance.

## 📂 Dataset

**Source**: Internal project dataset  
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

### Correlation Heatmap
- Strong relationships observed between `AQ`, `VOC`, and `fail`.

## 🤖 Models Used

### 1. **Support Vector Classifier (RBF Kernel)**
- Tuned using grid search on `C` and `gamma`.
- Excellent generalization and precision.

### 2. **Random Forest Classifier**
- Tuned over `max_depth` via grid search.
- Performed competitively with slight overfitting tendency.

## 📊 Model Performance

| Model           | Accuracy | Precision | Recall | F1 Score |
|----------------|----------|-----------|--------|----------|
| **SVC (RBF)**      | **0.894**   | **0.867**    | **0.895** | **0.881**   |
| Random Forest   | 0.887    | 0.859     | 0.887  | 0.873    |

## 🧩 Confusion Matrices

### SVC (RBF)
| Predicted | No Fail | Fail |
|-----------|---------|------|
| **No Fail** | 143     | 17   |
| **Fail**    | 13      | 111  |

### Random Forest
| Predicted | No Fail | Fail |
|-----------|---------|------|
| **No Fail** | 142     | 18   |
| **Fail**    | 14      | 110  |

## 🧠 Result and Interpretations

- **SVC with RBF** kernel performed slightly better than Random Forest across all metrics.
- It had higher **precision** and **F1 score**, making it more reliable for high-risk environments where false alarms are costly.
- Confusion matrices show both models are well-balanced, but SVC has fewer total misclassifications.

## ✅ Conclusion

SVC (RBF kernel) is selected as the final model due to its superior accuracy, better balance of precision and recall, and slightly fewer false predictions.  
It is well-suited for predictive maintenance systems where early detection of failure is critical.
