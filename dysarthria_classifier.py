import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def extract_features(audio_path):
    """Extract audio features using librosa."""
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, duration=3)
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Calculate statistics for each feature
        features = []
        for feature in [mfccs, chroma, mel, spectral_contrast]:
            features.extend([
                np.mean(feature),
                np.std(feature),
                np.max(feature),
                np.min(feature)
            ])
        
        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

def prepare_dataset(base_path):
    """Prepare the dataset from the audio files."""
    features = []
    labels = []
    
    # Process dysarthric samples
    for gender in ['female', 'male']:
        dys_path = os.path.join(base_path, f'dysarthria_{gender}')
        for file in os.listdir(dys_path):
            if file.endswith('.wav'):
                file_path = os.path.join(dys_path, file)
                feat = extract_features(file_path)
                if feat is not None:
                    features.append(feat)
                    labels.append(1)  # 1 for dysarthric
    
    # Process non-dysarthric samples
    for gender in ['female', 'male']:
        non_dys_path = os.path.join(base_path, f'non_dysarthria_{gender}')
        for file in os.listdir(non_dys_path):
            if file.endswith('.wav'):
                file_path = os.path.join(non_dys_path, file)
                feat = extract_features(file_path)
                if feat is not None:
                    features.append(feat)
                    labels.append(0)  # 0 for non-dysarthric
    
    return np.array(features), np.array(labels)

def train_model(base_path):
    """Train the random forest model."""
    # Prepare the dataset
    print("Extracting features from audio files...")
    X, y = prepare_dataset(base_path)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    print("Training Random Forest model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    train_score = clf.score(X_train_scaled, y_train)
    test_score = clf.score(X_test_scaled, y_test)
    print(f"Training accuracy: {train_score:.2f}")
    print(f"Testing accuracy: {test_score:.2f}")
    
    # Save the model and scaler
    print("Saving model and scaler...")
    with open('dysarthria_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return clf, scaler

def predict_audio(audio_path, model=None, scaler=None):
    """Predict whether an audio file is from a dysarthric patient."""
    if model is None or scaler is None:
        # Load the saved model and scaler
        with open('dysarthria_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    
    # Extract features from the audio file
    features = extract_features(audio_path)
    if features is None:
        return None
    
    # Scale the features and make prediction
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    return {
        'prediction': 'Dysarthric' if prediction == 1 else 'Non-dysarthric',
        'confidence': probability[prediction]
    }

if __name__ == '__main__':
    # Train the model
    base_path = 'torgo_data'
    model, scaler = train_model(base_path)
    
    # Example of how to use the prediction function
    # Replace with your audio file path
    # result = predict_audio('path_to_your_audio_file.wav')
    # print(f"Prediction: {result['prediction']}")
    # print(f"Confidence: {result['confidence']:.2f}")