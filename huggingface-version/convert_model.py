"""
Script untuk convert model dari pickle ke format yang lebih compatible
Jalankan: python convert_model.py
"""
import joblib
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Workaround numpy._core
import sys
if not hasattr(np, '_core'):
    try:
        import numpy.core as _core
        sys.modules['numpy._core'] = _core
    except:
        pass

def convert_model():
    """Convert model dari PKL format lama ke format baru yang compatible"""
    try:
        print("Loading original model...")
        model_package = joblib.load("svm_model_fixed.pkl")
        
        svm_model = model_package['model']
        scaler = model_package['scaler']
        pca = model_package['pca']
        
        print("✓ Model loaded successfully")
        
        # Save dengan protocol pickle yang lebih lama untuk compatibility
        print("Saving with compatible protocol...")
        new_package = {
            'model': svm_model,
            'scaler': scaler,
            'pca': pca
        }
        
        # Backup original
        if os.path.exists("svm_model_fixed.pkl"):
            os.rename("svm_model_fixed.pkl", "svm_model_fixed.pkl.bak")
            print("✓ Original model backed up as svm_model_fixed.pkl.bak")
        
        # Save dengan protocol lama (compatible dengan numpy versi lama)
        joblib.dump(new_package, "svm_model_fixed.pkl", protocol=4)
        print("✓ Model converted and saved successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = convert_model()
    if success:
        print("\nModel conversion complete! You can now push to Hugging Face.")
    else:
        print("\nConversion failed. Check the error above.")
