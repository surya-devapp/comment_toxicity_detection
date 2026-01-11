# Deep Learning Based Comment Toxicity Detection

## Project Overview
This project targets the automated detection of toxic comments in online discussions. Using Deep Learning (LSTM/Bi-LSTM), the model predicts the likelihood of toxicity in text. A Streamlit web application is provided for real-time interaction.

## Directories
- `data/`: Place your dataset here (e.g., `train.csv`).
- `src/`: Source code for preprocessing, modeling, and training.
- `models/`: Saved models and tokenizers.

## Setup Instructions
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Workflow
1. **Data Preparation**: The system expects a CSV with 'comment_text' and 'toxic' (target) columns.
2. **Training**: Run `python src/train.py` to train the model.
3. **Inference**: Use the Streamlit app for real-time predictions.

## Technical Details
- **Model**: Scikit-Learn MLPClassifier (Neural Network) with Calibration.
- **Feature Engineering**: TF-IDF + Sentiment Analysis + Target Detection (Hybrid Pipeline).
- **Preprocessing**: Tokenization, Stopword removal, Lemmatization.
- **Frameworks**: Scikit-Learn, Streamlit.

## Version History
See [MODEL_CHANGELOG.md](MODEL_CHANGELOG.md) for detailed version history.
- **v1**: Baseline (TF-IDF + MLP)
- **v2**: Calibrated Probabilities
- **v3**: Enhanced Features (Target Aware)
