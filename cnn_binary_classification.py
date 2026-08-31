import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = 64
N_SAMPLES = 600
rng = np.random.default_rng(42)

def make_circle_image(size, rng):
    img = rng.normal(0.1, 0.05, (size, size))
    cx = rng.integers(size // 4, 3 * size // 4)
    cy = rng.integers(size // 4, 3 * size // 4)
    r = rng.integers(size // 6, size // 3)
    yy, xx = np.ogrid[:size, :size]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    img[mask] = rng.normal(0.85, 0.05, mask.sum())
    return np.clip(img, 0, 1)

def make_square_image(size, rng):
    img = rng.normal(0.1, 0.05, (size, size))
    w = rng.integers(size // 4, size // 2)
    x0 = rng.integers(0, size - w)
    y0 = rng.integers(0, size - w)
    img[y0:y0 + w, x0:x0 + w] = rng.normal(0.85, 0.05, (w, w))
    return np.clip(img, 0, 1)

images = []
labels = []
for _ in range(N_SAMPLES // 2):
    images.append(make_circle_image(IMG_SIZE, rng))
    labels.append(0)
    images.append(make_square_image(IMG_SIZE, rng))
    labels.append(1)

X = np.array(images, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(labels, dtype=np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

model = keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    data_augmentation,
    layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("---- Manufacturing Quality Control CNN Example ----")
print("Class 0 = Round Component (Non-Defective) | Class 1 = Square Component (Defective Batch)")
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=32,
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.3f}")
print(f"Test loss: {test_loss:.3f}")

y_pred_probs = model.predict(X_test, verbose=0)
y_pred = (y_pred_probs >= 0.5).astype(int).flatten()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Non-Defective", "Defective"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

labels_map = {0: "Non-Defective", 1: "Defective"}
print("\nSample Predictions:")
for i in range(5):
    true_label = labels_map[int(y_test[i])]
    pred_label = labels_map[int(y_pred[i])]
    confidence = y_pred_probs[i][0] if y_pred[i] == 1 else 1 - y_pred_probs[i][0]
    print(f"Sample {i+1}: True = {true_label} | Predicted = {pred_label} | Confidence = {confidence:.3f}")

new_image = make_square_image(IMG_SIZE, rng).reshape(1, IMG_SIZE, IMG_SIZE, 1)
new_pred = model.predict(new_image, verbose=0)[0][0]
new_label = labels_map[int(new_pred >= 0.5)]
print(f"\nNew Product Scan -> Predicted: {new_label} (Confidence: {new_pred if new_pred >= 0.5 else 1 - new_pred:.3f})")
