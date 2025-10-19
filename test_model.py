import tensorflow as tf
import os

cotton_path = "models/cottonbest.keras"
wheat_path   = "models/best_model.keras"
corn_path = "models/cornbest.keras"
rice_path = "models/rice_best_model.keras"
crops_classifier_path = "models/crops_classifier.keras"

print("TensorFlow version:", tf.__version__)
print("\nChecking file existence:")

print("Cotton:", os.path.exists(cotton_path), cotton_path)
print("Wheat  :", os.path.exists(wheat_path), wheat_path)
print("Corn  :", os.path.exists(corn_path), corn_path)
print("Rice  :", os.path.exists(rice_path), rice_path)
print("CropClassifier  :", os.path.exists(crops_classifier_path), crops_classifier_path)


# Load cotton model
try:
    cotton_model = tf.keras.models.load_model(cotton_path, compile=False)
    print("\n✅ Cotton model loaded successfully")
    print("Input shape:", cotton_model.input_shape)
except Exception as e:
    print("\n❌ Cotton model load failed:", e)

# Load wheat model
try:
    wheat_model = tf.keras.models.load_model(wheat_path, compile=False)
    print("\n✅ wheat model loaded successfully")
    print("Input shape:", wheat_model.input_shape)
except Exception as e:
    print("\n❌ wheat model load failed:", e)

    # Load corn model
try:
    corn_model = tf.keras.models.load_model(corn_path, compile=False)
    print("\n✅ Corn model loaded successfully")
    print("Input shape:", corn_model.input_shape)
except Exception as e:
    print("\n❌ Corn model load failed:", e)


    # Load rice model
try:
    rice_model = tf.keras.models.load_model(rice_path, compile=False)
    print("\n✅ Rice model loaded successfully")
    print("Input shape:", rice_model.input_shape)
except Exception as e:
    print("\n❌ Rice model load failed:", e)


# Load Crops Classifier model
try:
    crops_classifier_model = tf.keras.models.load_model(crops_classifier_path, compile=False)
    print("\n✅ Crops Classifier model loaded successfully")
    print("Input shape:", crops_classifier_model.input_shape)
except Exception as e:
    print("\n❌ Crops Classifier model load failed:", e)