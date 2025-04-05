#=================================================================================================================
#                                           Libraries
#=================================================================================================================
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
import tensorflow as tf
#=================================================================================================================
#                                           Image Preparation
#=================================================================================================================
def load_custom_dataset(folder_path):
    image_data = []
    labels = []

    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)

        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                
                
                if image_file.endswith(".jpg"):
                    img = Image.open(image_path)
                    
                    # image -> NumPy array
                    img_array = np.array(img)
                    
                    # Add label to image data
                    image_data.append(img_array)
                    
                    labels.append(int(label))  # folder names
                    
    return np.array(image_data), np.array(labels)

def preprocess_image(image_data):# Convert each 28x28 matrix to grayscale and resize to 8x8
    preprocessed_images = []

    for img_array in image_data:
        # Convert the 28x28 matrix to PIL Image
        img = Image.fromarray(img_array)

        # Convert the image to grayscale
        img_gray = img.convert("L")

        # Resize the image to 8x8 pixels
        img_resized = img_gray.resize((8, 8))
    
        # Convert the resized image to a NumPy array
        img_array_resized = np.array(img_resized)

        preprocessed_images.append(img_array_resized)

    return np.array(preprocessed_images)
#=================================================================================================================
#                                           Loading Dataset
#=================================================================================================================
dataset_path = ".\Dataset\Train"
images, labels = load_custom_dataset(dataset_path)
preprocessed_images = preprocess_image(images)

X_train, X_test, y_train, y_test = train_test_split(preprocessed_images, labels, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()

# Reshape the 8x8 images to 1D arrays
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

X_train_scaled = scaler.fit_transform(X_train_flattened)
X_test_scaled = scaler.transform(X_test_flattened)


#=================================================================================================================
#                                           Implement Your Algorithm
#=================================================================================================================
# Create a KNN classifier
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the KNN model
knn_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_test_pred = knn_model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy on Test Set: {accuracy * 100:.2f}%")


#=================================================================================================================
#                                           NEURAL NETWORK (BONUS)
#=================================================================================================================
# Convert labels to one-hot encoding: each class is represented as a vector with all zeros except for the index corresponding to the class, which is set to 1
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(64,)), # ReLU: activation function: f(x) = max(0, x)
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax') # output layer: it's a classification problem with 10 classes
    #softmax(x)_i = exp(x_i) / sum(exp(x_j)) for all j
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model for 10 epochs
history = model.fit(X_train_scaled, y_train_onehot, epochs=10, validation_data=(X_test_scaled, y_test_onehot))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_onehot)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

#=================================================================================================================
#                                           Test Section
#=================================================================================================================

# Leave it blank

#=================================================================================================================
#                                           Test one picture
#=================================================================================================================
def preprocess_single_image(image_path):
    img = Image.open(image_path)
    img_gray = img.convert("L")  # Convert to grayscale
    img_resized = img_gray.resize((8, 8))  # Resize to 8x8
    img_array = np.array(img_resized).reshape(1, -1)  # Flatten to 1D array
    return img_array


image_path = '.\Test\image_10000.jpg'
ActualLabel = 5
preprocessed_image_test = preprocess_single_image(image_path)

# preprocessed_image_test_scaled = scaler.fit_transform(preprocessed_image_test)

print("Actual Label =", ActualLabel)

knn_prediction = knn_model.predict(preprocessed_image_test)
print(f"KNN Prediction: {knn_prediction[0]}")

nn_prediction = model.predict(preprocessed_image_test)
nn_predicted_label = np.argmax(nn_prediction)
print(f"Neural Network Prediction: {nn_predicted_label}")

#=================================================================================================================
#                                           Visualization
#=================================================================================================================
random_index = random.randint(0, len(X_test) - 1)


selected_image = X_test[random_index]
actual_label = y_test[random_index]
predicted_label = y_test_pred[random_index]


plt.imshow(selected_image, cmap='gray')
plt.title(f"Actual Label: {actual_label}, Predicted Label: {predicted_label}")
plt.show()
############################################################## NEURAL NETWORK
# Plot accuracy and loss for each epoch
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
