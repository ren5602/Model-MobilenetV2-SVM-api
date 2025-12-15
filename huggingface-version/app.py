from flask import Flask, request, render_template_string
import numpy as np
import sys
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import joblib
import os
from io import BytesIO

# Fix numpy._core compatibility issue
if not hasattr(np, '_core'):
    try:
        import numpy.core as _core
        sys.modules['numpy._core'] = _core
    except (ImportError, AttributeError):
        pass

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

@app.route('/', methods=['GET'])
def home():
    return render_template_string('''
    <div style="text-align:center; padding:50px;">
        <h1>🥬 Vegetable Cleanliness Classifier</h1>
        <p>Prediksi apakah sayuran Anda dalam kondisi <b>bersih</b> atau <b>kotor</b></p>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required><br><br>
            <button type="submit" style="padding:10px 20px; font-size:16px;">Prediksi</button>
        </form>
    </div>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return "<h2 style='text-align:center; color:red;'>Error: Tidak ada file yang dikirim</h2><center><a href='/'>Kembali</a></center>"
        
        file = request.files['file']
        
        if file.filename == '':
            return "<h2 style='text-align:center; color:red;'>Error: File tidak dipilih</h2><center><a href='/'>Kembali</a></center>"
        
        # Simpan file sementara
        file_path = os.path.join('temp_image.jpg')
        file.save(file_path)
        
        # Ekstraksi fitur dari gambar
        features = extract_feature_single(file_path)
        
        # Normalisasi fitur menggunakan scaler
        features_scaled = scaler.transform(features)
        
        # Reduksi dimensi menggunakan PCA
        features_pca = pca.transform(features_scaled)

        # Prediksi dengan model SVM
        prediction = svm_model.predict(features_pca)[0]
        predicted_index = int(prediction)
        predicted_label = class_labels[predicted_index]
        
        # Hitung confidence jika tersedia
        confidence_text = ""
        if hasattr(svm_model, "decision_function"):
            decision_values = svm_model.decision_function(features_pca)[0]
            confidence = float(abs(decision_values))
            # Konversi ke persentase menggunakan sigmoid function
            confidence_percentage = (1 / (1 + np.exp(-confidence))) * 100
            confidence_text = f"<p>Tingkat Kepercayaan: {confidence_percentage:.2f}%</p>"
        
        # Tentukan warna berdasarkan hasil
        color = "green" if predicted_label == "bersih" else "red"
        
        html_response = f'''
        <div style="text-align:center; padding:50px;">
            <h2 style="color:{color};">Hasil Prediksi: {predicted_label}</h2>
            {confidence_text}
            <center><a href='/' style="padding:10px 20px; font-size:14px; text-decoration:none; background-color:#007bff; color:white; border-radius:5px;">Kembali</a></center>
        </div>
        '''
        
        return html_response
    
    except Exception as e:
        return f"<h2 style='text-align:center; color:red;'>Error: {str(e)}</h2><center><a href='/'>Kembali</a></center>"
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    # Port 7860 wajib untuk Hugging Face Spaces
    app.run(host='0.0.0.0', port=7860)