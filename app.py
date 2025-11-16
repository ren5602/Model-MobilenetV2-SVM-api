from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import joblib
import os

# Inisialisasi Flask
app = Flask(__name__)

# Load model SVM
model_package = joblib.load("svm_model_fixed.pkl")
svm_model = model_package['model']
scaler = model_package['scaler']
pca = model_package['pca']

# Load MobileNetV2 tanpa fully connected layer
mobilenet = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(128,128,3))

# Label kelas 
class_labels = ["bersih", "kotor"]

def extract_feature_single(img_path):
    """Ekstraksi fitur satu gambar (ukuran 128x128)"""
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # normalisasi MobileNetV2
    features = mobilenet.predict(img_array, verbose=0)
    return features.reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar dikirim'}), 400

    file = request.files['image']
    file_path = os.path.join('temp_image.jpg')
    file.save(file_path)

    try:
        # Ekstraksi fitur dari gambar
        features = extract_feature_single(file_path)
        
        # Normalisasi fitur menggunakan scaler
        features_scaled = scaler.transform(features)
        
        # Reduksi dimensi menggunakan PCA
        features_pca = pca.transform(features_scaled)

        # Prediksi dengan model SVM
        prediction = svm_model.predict(features_pca)[0]
        probabilities = None
        if hasattr(svm_model, "decision_function"):
            probabilities = svm_model.decision_function(features_pca).tolist()
        
        predicted_index = int(prediction)
        predicted_label = class_labels[predicted_index]

        return jsonify({
            'predicted_label': predicted_label,
            'predicted_index': predicted_index,
            # 'confidence_score': probabilities[0] if probabilities else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)