import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

rf_model = joblib.load("./models/myModel.pkl")

def extract_mfcc(audio_file, max_length=100):
    audiofile, sr = librosa.load(audio_file)
    fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=20)
    if fingerprint.shape[1] < max_length:
        pad_width = max_length - fingerprint.shape[1]
        fingerprint_padded = np.pad(fingerprint, pad_width=((0, 0), (0, pad_width)), mode='constant')
        return fingerprint_padded.T
    elif fingerprint.shape[1] > max_length:
        return fingerprint[:, :max_length].T
    else:
        return fingerprint.T

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        
        if file.filename == '':
            return 'No selected file'
        
        if file:
            file_path = os.path.join('./uploads', file.filename)
            file.save(file_path)
            
            mfcc_features = extract_mfcc(file_path)
            mfcc_features_flat = mfcc_features.flatten()
            
            prediction = rf_model.predict([mfcc_features_flat])[0]
            
            os.remove(file_path)
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    if not os.path.exists('./uploads'):
        os.makedirs('./uploads')
    app.run(debug=True)
