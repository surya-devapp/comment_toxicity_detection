# Deep Learning for Comment Toxicity Detection with Streamlit

## Domain: Online Community Management and Content Moderation

### Skills Takeaway
- Deep Learning & Neural Networks
- Model Development and Training (PyTorch)
- Model Evaluation and Optimization (F2-score, Recall)
- Streamlit Web App Development
- Model Deployment & NLP Preprocessing

This project implements a real-time toxicity detection system for online comments using a Deep Learning model (Bi-LSTM) built with PyTorch and deployed via Streamlit.

## Features
- **Real-time Prediction**: Analyze individual comments for toxicity.
- **Multi-label Classification**: Detects 6 types of toxicity (toxic, severe toxic, obscene, threat, insult, identity hate).
- **Bulk Prediction**: Upload a CSV file to analyze multiple comments at once.
- **Interactive Dashboard**: Visualizes toxicity levels with interactive charts.

## Technologies Used
- **Python**: Core programming language.
- **PyTorch**: Deep Learning framework for model training and inference.
- **Streamlit**: Web application framework for the user interface.
- **Pandas/Numpy**: Data manipulation.
- **Plotly**: Interactive visualizations.

## Project Structure
- `app.py`: Main Streamlit application.
- `model.py`: Bi-LSTM model architecture.
- `preprocessing.py`: Text cleaning and tokenization logic.
- `train.py`: Model training script.
- `utils.py`: Inference helper functions.
- `data/`: Dataset storage.
- `models/`: Trained model and tokenizer storage.

## 🚀 Deployment Guide
Detailed instructions for setting up and deploying the application.

### Local Deployment
1. Ensure Python 3.8+ is installed.
2. Clone the repository and navigate to the project directory.
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
5. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### ☁️ Cloud Deployment (Streamlit Sharing)
1. Push the code to a public GitHub repository.
2. Log in to [Streamlit Cloud](https://share.streamlit.io/).
3. Click "New App", select your repository, branch, and `app.py` as the main file path.
4. Click "Deploy".

## 🤖 Moderation Workflow (HITL)
To ensure ethical and accurate moderation, the application uses a **Human-in-the-Loop (HITL)** strategy for borderline cases (scores between 0.4 and 0.7):

1.  🟢 **Safe**: Content is benign.
    - *Note*: If labeled **Safe (Needs Human Review)**, manual verification is recommended.
2.  🔴 **Toxic**: Content violates community standards.
    - *Note*: If labeled **Toxic (Needs Human Review)**, human judgment should verify the AI's decision to avoid false bans.

## Performance (Phase 2 Optimized)
The model has been optimized using a **Balanced Sampling** strategy to address class imbalance, resulting in a significantly higher recall for toxic comments.

| Metric | Value |
| :--- | :--- |
| **Weighted F2-score (beta=2)** | **0.7600** |
| **Toxic Recall** | **0.95** |
| **Insult Recall** | **0.91** |
| **Obscene Recall** | **0.86** |
| **Sensitivity Optimization** | Deeper 128-unit Bi-LSTM, Weighted Loss (`BCEWithLogitsLoss`), and optimized sampling. |

*Optimization: Trained for 6 epochs with best-model checkpointing and balanced minority class representation.*

## Dataset
The project uses the Jigsaw Toxic Comment Classification Challenge dataset from Wikipedia talk page edits.
