```markdown
# Dysarthria Predictor

A speech-based dysarthria detection project that implements preprocessing, feature extraction, and machine learning models to classify dysarthria from audio recordings.

## Overview
This repository contains code to:
- Load and preprocess speech audio data
- Extract acoustic features (e.g., MFCCs, spectrograms)
- Train and evaluate machine learning models to detect dysarthria
- Provide scripts for inference on new audio samples

## Features
- Configurable preprocessing pipeline (resampling, trimming, noise reduction)
- Feature extraction modules (MFCCs, delta features, spectrogram)
- Training scripts with cross-validation and performance metrics
- Example inference script for single audio files

## Getting Started

### Requirements
- Python 3.8+
- Common libraries: numpy, scipy, librosa, scikit-learn, pandas, torch (if using deep models)
- See `requirements.txt` for full list.

### Installation
1. Clone the repository:
   ```
   git clone https://github.com/theadithyankr/Dysarthria_predictor.git
   cd Dysarthria_predictor
   ```
2. Create virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Usage
- Preprocess data:
  ```
  python scripts/preprocess.py --input-dir data/raw --output-dir data/processed
  ```
- Extract features:
  ```
  python scripts/extract_features.py --input-dir data/processed --output features/features.npy
  ```
- Train model:
  ```
  python scripts/train.py --features features/features.npy --model-dir models/
  ```
- Inference on a single audio file:
  ```
  python scripts/infer.py --model models/best.pt --audio example.wav
  ```

## Data
Provide or link to the dataset you used (ensure compliance with licensing and privacy). Example: "Dataset: [link or description]"

## Evaluation
Include metrics used (accuracy, precision, recall, F1, ROC AUC) and sample results or a link to experiment logs.

## Contributing
Contributions welcome. Please open issues or pull requests with suggested improvements.

## License
Specify the license (e.g., MIT). If none, add a LICENSE file.

## Contact
Adithyan K — GitHub: @theadithyankr — email: theadhithyankr@gmail.com
```
