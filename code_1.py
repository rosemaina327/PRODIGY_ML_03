import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Function to load and preprocess images
def load_images_from_folder(folder, label, size=(32, 32)):  # Reduced size to 32x32
    images, labels = [], []
    corrupt_files = []  # To track corrupt files
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            with open(img_path, 'rb') as f:
                file_bytes = f.read()
                if file_bytes[:2] != b'\xff\xd8':  # Check for JPEG file signature
                    corrupt_files.append(filename)
                    continue
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            if img is not None:
                img = cv2.resize(img, size)  # Resize to 32x32
                images.append(img.flatten())  # Flatten the image
                labels.append(label)  # Append label (0 for cat, 1 for dog)
            else:
                corrupt_files.append(filename)
        except Exception as e:
            corrupt_files.append(filename)
    if corrupt_files:
        print(f"Skipped {len(corrupt_files)} corrupt files: {corrupt_files}")
    return images, labels

# Define folder paths
cat_folder = r"D:/Vs Code tasks/ML/SVM/PetImages/Cat"
dog_folder = r"D:/Vs Code tasks/ML/SVM/PetImages/Dog"

# Load and preprocess the images
print("Loading images...")
cat_images, cat_labels = load_images_from_folder(cat_folder, label=0)
dog_images, dog_labels = load_images_from_folder(dog_folder, label=1)

# Combine cat and dog data
X = np.array(cat_images + dog_images)
y = np.array(cat_labels + dog_labels)

print("Total samples:", len(X))
print("Labels distribution:", np.bincount(y))

# Split data into training and test sets
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set size:", len(X_train))
print("Test set size:", len(X_test))

# Apply PCA for dimensionality reduction
print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=300)  # Reduce to 300 principal components
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train an SVM model
print("Starting model training...")
svm_model = SVC(kernel='linear', C=1.0, cache_size=500, verbose=True)
svm_model.fit(X_train_pca, y_train)
print("Model training complete.")

# Evaluate the model
print("Evaluating model...")
y_pred = svm_model.predict(X_test_pca)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Test with a new image
image_path = r"C:\Users\maina\Pictures\test_image.jpg"  # Update this with your test image path
try:
    print("Testing with a new image...")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (32, 32))  # Match reduced size
    img_flattened = img_resized.flatten().reshape(1, -1)
    img_pca = pca.transform(img_flattened)  # Apply PCA to match training data
    prediction = svm_model.predict(img_pca)
    print("Prediction:", "Dog" if prediction[0] == 1 else "Cat")
except Exception as e:
    print("Error testing with new image:", e)
