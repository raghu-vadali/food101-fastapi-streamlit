""" import following packages before using"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import zipfile
import requests
import random
import os
import datetime
import numpy as np
import seaborn as sns
from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
confusion_matrix,
classification_report
)


# Regression metrics
def regression_metrics(y_test, y_pred):
    """ 
    y_test = Y values of testing data
    y_pred = Y predicted values
    """
    mae = tf.keras.losses.MeanAbsoluteError()
    mse = tf.keras.losses.MeanSquaredError()
    mape = tf.keras.losses.MeanAbsolutePercentageError()
    msle = tf.keras.losses.MeanSquaredLogarithmicError()
    cos_sim = tf.keras.losses.CosineSimilarity()
    huber = tf.keras.losses.Huber()

    print(f"MAE  : {mae(y_test, y_pred).numpy():.2f}")
    print(f"MSE  : {mse(y_test, y_pred).numpy():.2f}")
    print(f"MAPE : {mape(y_test, y_pred).numpy():.2f}")
    print(f"MSLE : {msle(y_test, y_pred).numpy():.2f}")
    print(f"Cosine Similarity : {cos_sim(y_test, y_pred).numpy():.2f}")
    print(f"Huber Loss: {huber(y_test, y_pred).numpy():.2f}")
    
# plot decision boundary for classification
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = (Z > 0.5).astype(int)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=10)
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.title('Decision Boundary')
    plt.show()
    
# Binary Classification metrics
def classification_metrics(y_test, y_pred):
    """ 
    y_test = Y values of testing data
    y_pred = Y predicted values
    """
        
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)
    
    # Print results
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", cm)
    sns.heatmap(cm, annot=True, fmt='d')
    print("Classification Report:\n", report)

def Multi_classification_metrics(y_test, y_pred):
    """
    Multiclass classification metrics.
    y_test: true labels (1D array of ints)
    y_pred: predicted labels (1D array of ints)
    """

    # If predictions are probs, take argmax
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)

    # Metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall    = recall_score(y_test, y_pred, average="weighted")
    f1        = f1_score(y_test, y_pred, average="weighted")
    cm        = confusion_matrix(y_test, y_pred)
    report    = classification_report(y_test, y_pred)

    # Print results
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("Classification Report:\n", report)

    # Confusion matrix heatmap
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report,
        "cm": cm
    }
    
# Training loss vs accuracy
def loss_acc(model_fitted):
    """ model_fitted = model already fitted """
    plt.plot(model_fitted.history["loss"], label="loss")
    plt.plot(model_fitted.history["accuracy"], label="accuracy")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()

# learning rate vs accuracy
def lr_acc(model_fitted):
    """ model_fitted = model already fitted """
    plt.plot(model_fitted.history["learning_rate"], model_fitted.history["loss"])
    plt.title("Training Learning rate and Loss")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    
    
# Image visualization
def select_random_image(target_dir, target_class_list):
    """ target_dir = directory path
        target_class_list = target class list """
    target_class  = random.sample(target_class_list, 1)[0]
    target_folder = target_dir+target_class                    # Setup target directory (we'll view images from here)
    random_image = random.sample(os.listdir(target_folder), 1) # get a random selection of images as list, here we gave only 1 image
    img = mpimg.imread(target_folder + "/" + random_image[0])  # Read in the image and plot it using matplotlib   
    plt.imshow(img)
    plt.axis(False)
    return target_class, random_image[0], img

# Image visualization and Augumentation effect
def image_augumentation(target_dir, target_class_list, imagedata_augumentation):
    print(" Gives only random feature modification of image from defined Augumentation in ImageDataGenerator")
    print("""all defined augmentations are applied to every image. \n
           Instead, each image gets a random subset of the augmentations \n
           based on the ranges and probabilities one has specified""")

    class_name, img_name, img = select_random_image(target_dir, target_class_list)
    plt.close('all') 
    print(f"Class: {class_name}\nImage Name: {img_name}")
    print(f"Image shape: {img.shape}") # show the shape of the image
    
    # Normal image
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis(False);

    # Normal image + augmented
    plt.subplot(1,2,2)
    plt.imshow(tf.keras.utils.array_to_img(imagedata_augumentation(img))) # apply a random transformation to the image
    plt.title("Augumented Image")
    plt.axis(False);
    
    plt.tight_layout()

# plotting loss and accuracy curves
def plot_loss_accuracy(fitted_model):
    history = fitted_model.history

    # Plot Loss
    plt.figure(figsize=(14, 10))
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Model Loss")
    plt.ylim(0, 3)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history["accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Val Accuracy")
    plt.title("Model Accuracy")
    plt.ylim(0, 3)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    
    # Plot training accuracy and loss
    plt.subplot(2, 2, 3)
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["accuracy"], label="Train accuracy")
    plt.title("Model Train accuracy and Loss")
    plt.ylim(0, 3)
    plt.xlabel("Epochs")
    plt.ylabel("aacuracy/Loss")
    plt.legend()
    plt.grid(True)
    
    # Plot validation accuracy and loss
    plt.subplot(2, 2, 4)
    plt.plot(history["val_loss"], label="Train Loss")
    plt.plot(history["val_accuracy"], label="Train accuracy")
    plt.title("Model validation accuracy and Loss")
    plt.ylim(0, 3)
    plt.xlabel("Epochs")
    plt.ylabel("aacuracy/Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
# Download image data and unzip
def download_and_unzip(url):
           
    # Get filename from URL
    filename = url.split("/")[-1]
    
    # Get the current file directory
    script_dir = os.path.dirname(os.path.abspath(__file__))    
    zip_path = os.path.join(script_dir, filename)
    print(script_dir)

    # Download the file
    response = requests.get(url)
    with open(zip_path, "wb") as f:
        f.write(response.content)

    # Unzip the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(script_dir)

    folder_name = filename.replace(".zip", "")
    print(f"Downloaded and extracted: {folder_name} into {script_dir} at {datetime.datetime.now()}")
    
    return folder_name

def walk_through_dir(folder_name):
    for dirpath, dirnames, filenames in os.walk(folder_name):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}")
    return dirpath, dirnames

def get_image_count(target_dir):
    count = 0
    for dirpath, dirnames, filenames in os.walk(target_dir):
        count += len(filenames)
    return count

# callback function to be written on tensorboard
def create_tensorboard_callback(dir_name, experiment_name):
    base_dir = os.path.abspath(dir_name)
    exp_dir = os.path.join(base_dir, experiment_name)
    log_dir = os.path.join(
        exp_dir,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    # 🔒 Force-clean any file collisions
    for path in [base_dir, exp_dir]:
        if os.path.exists(path) and os.path.isfile(path):
            os.remove(path)

    # ✅ Ensure directories exist
    os.makedirs(log_dir, exist_ok=True)

    print(f"Saving TensorBoard log files to: {log_dir}")

    return tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )

def preprocess_image(img_path, target_size=(224, 224)):
    
    """Load and preprocess image for prediction."""
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)    
    
    return np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)

def predict_image(model, img_path, class_names=None):
    
    """Predict class probabilities and return top result."""
    img_resize = preprocess_image(img_path)
    probs = model.predict(img_resize)  # shape: (101,)
    top_idx = np.argmax(probs[0])            # - Finds the index of the maximum probability
    top_prob = probs[0][top_idx]
    
    if class_names:
        print(f"Predicted class: {class_names[top_idx]} with probability {top_prob:.4f}")
    else:
        print(f"Predicted class index: {top_idx} with probability {top_prob:.4f}")
    
    return probs

    