# 🧠 Handwritten Digit Classification Using Artificial Neural Network (ANN)

## 📌 Project Overview

This project implements a Handwritten Digit Classification system using a Deep Neural Network (ANN) built with TensorFlow and Keras. The model is trained on the MNIST dataset to recognize handwritten digits from 0 to 9.

The project demonstrates the complete machine learning workflow including data preprocessing, model training, evaluation, and prediction.

---

## 📂 Dataset

The project uses the MNIST dataset, which contains:

* 60,000 training images
* 10,000 testing images
* Image size: 28 × 28 pixels
* Grayscale images
* 10 output classes (digits 0–9)

Dataset source: TensorFlow/Keras built-in dataset.

---

## 🔧 Data Preprocessing

### 1. Load Dataset

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```

### 2. Normalize Pixel Values

Pixel values are scaled from 0–255 to 0–1.

```python
x_train = x_train / 255.0
x_test = x_test / 255.0
```

---

## 🏗️ Model Architecture

The model is a fully connected Artificial Neural Network (ANN).

```python
model = keras.Sequential([
    keras.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

### Architecture Details

| Layer       | Description                                        |
| ----------- | -------------------------------------------------- |
| Input Layer | Accepts 28×28 grayscale images                     |
| Flatten     | Converts 28×28 image into a 784-dimensional vector |
| Dense (128) | Hidden layer with ReLU activation                  |
| Dense (64)  | Hidden layer with ReLU activation                  |
| Dense (10)  | Output layer with Softmax activation               |

---

## ⚙️ Training Configuration

```python
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_split=0.2,
    batch_size=32
)
```

| Parameter        | Value                           |
| ---------------- | ------------------------------- |
| Optimizer        | Adam                            |
| Loss Function    | Sparse Categorical Crossentropy |
| Epochs           | 10                              |
| Batch Size       | 32                              |
| Validation Split | 20%                             |

---

## 📊 Model Performance

### Test Results

| Metric             | Score |
| ------------------ | ----- |
| Accuracy           | 98%   |
| Macro Precision    | 0.98  |
| Macro Recall       | 0.98  |
| Macro F1-Score     | 0.98  |
| Weighted Precision | 0.98  |
| Weighted Recall    | 0.98  |
| Weighted F1-Score  | 0.98  |

### Classification Report

```text
Accuracy: 0.98

Macro Avg:
Precision: 0.98
Recall: 0.98
F1-Score: 0.98

Weighted Avg:
Precision: 0.98
Recall: 0.98
F1-Score: 0.98
```

---

## 📈 Confusion Matrix

A confusion matrix was generated to evaluate model performance and identify classification errors across different digit classes.

The confusion matrix provides insights into:

* Correct predictions
* Misclassified digits
* Class-wise performance
* Model weaknesses

---

## 🔮 Prediction Example

```python
predictions = model.predict(x_test)

predicted_digit = np.argmax(predictions[0])

print("Predicted:", predicted_digit)
print("Actual:", y_test[0])
```

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Scikit-learn

---

## 🚀 Results

✅ Successfully classified handwritten digits using a Deep Neural Network (ANN)

✅ Achieved approximately 98% test accuracy

✅ Generated confusion matrix and classification report for performance evaluation

✅ Demonstrated image preprocessing and multi-class classification techniques

---

## 📌 Future Improvements

* Implement Convolutional Neural Networks (CNNs)
* Hyperparameter tuning
* Data augmentation
* Model deployment using Streamlit
* Improve accuracy beyond 99%

---

## 👨‍💻 Author

Machine Learning project developed using TensorFlow and Keras for handwritten digit recognition on the MNIST dataset.
