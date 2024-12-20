# Cats vs. Dogs Classification using SVM

This project involves building a Support Vector Machine (SVM) model to classify images of cats and dogs. The focus is on preprocessing image data, training the SVM model, evaluating its performance, and making predictions.

---

## Objective:
The goal is to implement an SVM to classify images of cats and dogs using the Kaggle *Cats vs. Dogs* dataset. Through this project, you’ll learn how to preprocess image data and apply machine learning techniques effectively.

---

## Data:
- **Dataset**: Kaggle's *Cats vs. Dogs* dataset.
- **Images**: Cats (`0`) and dogs (`1`).
- **Preprocessing**: Resize to 64x64 pixels, convert to grayscale, and flatten images.

---

## Workflow:
### 1. Data Preprocessing:
- Handle corrupted files during loading.
- Resize images to 64x64 pixels.
- Convert to grayscale and flatten into 1D arrays.

### 2. Train-Test Split:
- Split data into 80% training and 20% testing sets.

### 3. Model Training:
- Train an SVM model with a linear kernel:
  ```python
  from sklearn.svm import SVC
  model = SVC(kernel='linear', C=1.0)
  model.fit(X_train, y_train)

### 4. Model Evaluation:
- Test the model:
  ```python
  from sklearn.metrics import accuracy_score ,classification_report
  y_pred = model.predict(X_test)
  print("Accuracy:", accuracy_score(y_test, y_pred))
  print("Classification Report:\n", classification_report(y_test, y_pred))



### 5.Testing with New Images:
- Predict the class of images
 ```python
 prediction = model.predict(new_image)
 print("Prediction:", "Dog" if prediction[0] == 1 else "Cat")
 ```


### 6. Output:
 - Accuracy: Measures overall model performance on the test set.
 -Classification Report: Precision, recall, and F1-score for cats and dogs.
 - Predictions: Classify whether a new image is a cat or a dog.

### 7.TOOLS:
- Python Libraries: numpy, scikit-learn, opencv-python.
- Environment: VS code

### 8. Learnings:
- Preprocessing raw image data for machine learning.
- Training and tuning SVM models for binary classification.
- Evaluating models using performance metrics.

### 9. Challenges:
- Handling corrupted image files.
- Balancing the trade-off between simplicity and performance in SVM.











