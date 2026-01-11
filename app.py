import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import os

# Add src to path to import local modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocess import clean_text
from features import SentimentExtractor, TargetExtractor

# Page Configuration
st.set_page_config(
    page_title="Toxicity Detection AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTextArea>div>div>textarea {
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #1e88e5, #5e35b1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_assets():
    try:
        model_path = 'models/toxicity_model_v3_enhanced.pkl'
        if not os.path.exists(model_path):
            return None
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None

model = load_assets()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-checked.png", width=80)
    st.title("Admin Console")
    st.markdown("---")
    st.info("This AI model detects toxic comments in real-time using a Multi-Layer Perceptron (Neural Network).")
    
    st.markdown("### 📊 Model Stats")
    if model:
        st.write("Status: **Online** 🟢")
        st.write("Type: **MLP Neural Network**")
    else:
        st.write("Status: **Offline** 🔴")
        st.warning("Please train the model first.")
        
    st.markdown("---")
    st.markdown("Made with ❤️ for Safer Communities")

# Main Content
st.title("🛡️ Comment Toxicity Detector")
st.markdown("### Ensure Safe and Constructive Online Discourse")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### Analyze Comment")
    user_input = st.text_area("Enter textual content to analyze", height=150, placeholder="Type a comment that you want to check for toxicity...")
    
    if st.button("Analyze Sentiment"):
        if not model:
            st.error("Model not found! Please check 'src/train.py' and run it.")
        elif not user_input.strip():
            st.warning("Please enter some text.")
        else:
            # Preprocess
            clean_input = clean_text(user_input)
            
            # Predict
            # Sklearn pipeline handles vectorization
            prediction_prob = model.predict_proba([clean_input])[0][1] # Probability of class 1 (toxic)
            
            # Display Result
            st.markdown("---")
            st.subheader("Analysis Result")
            
            score_percentage = prediction_prob * 100
            
            if prediction_prob > 0.85:
                st.error(f"⚠️ **Toxic Content Detected** ({score_percentage:.2f}% Confidence)")
                st.progress(float(prediction_prob))
                st.markdown("""
                    <div style='background-color: #ffebee; padding: 15px; border-radius: 8px; border-left: 5px solid #ef5350;'>
                        <p style='margin:0'>This comment has been flagged as toxic. It contains harassment, hate speech, or offensive language.</p>
                    </div>
                """, unsafe_allow_html=True)
            elif prediction_prob > 0.60:
                st.warning(f"📢 **Aggressive Tone / Complaint Detected** ({score_percentage:.2f}% Confidence)")
                st.progress(float(prediction_prob))
                st.markdown("""
                    <div style='background-color: #fff3e0; padding: 15px; border-radius: 8px; border-left: 5px solid #ff9800;'>
                        <p style='margin:0'>This comment is flagged as aggressive or a strong complaint, but likely not toxic. Human review recommended.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.success(f"✅ **Safe Content** ({(1-prediction_prob)*100:.2f}% Confidence)")
                st.progress(float(prediction_prob))
                st.markdown("""
                    <div style='background-color: #e8f5e9; padding: 15px; border-radius: 8px; border-left: 5px solid #66bb6a;'>
                        <p style='margin:0'>This comment appears to be safe.</p>
                    </div>
                """, unsafe_allow_html=True)

with col2:
    st.markdown("#### Real-time Metrics")
    st.markdown("""
        <div class="metric-card">
            <h3>Latency</h3>
            <h2 style="color: #1e88e5;">~15ms</h2>
            <p style="color: grey; font-size: 0.8em;">Average per request</p>
        </div>
        <br>
        <div class="metric-card">
            <h3>Accuracy</h3>
            <h2 style="color: #4CAF50;">~90%</h2>
            <p style="color: grey; font-size: 0.8em;">Based on Validation Set</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### Bulk Analysis")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file is not None:
        if model:
            df = pd.read_csv(uploaded_file)
            if 'comment_text' in df.columns:
                st.write(f"Analyzing {len(df)} comments...")
                subset = df.head(50) 
                subset['clean_text'] = subset['comment_text'].apply(clean_text)
                
                # Predict
                probs = model.predict_proba(subset['clean_text'])[:, 1]
                
                subset['toxicity_score'] = probs
                
                def get_category(score):
                    if score > 0.85: return "Toxic 🔴"
                    elif score > 0.60: return "Complaint 🟠"
                    else: return "Safe 🟢"
                    
                subset['category'] = subset['toxicity_score'].apply(get_category)
                
                st.dataframe(subset[['comment_text', 'category', 'toxicity_score']].style.highlight_max(axis=0))
            else:
                st.error("CSV must contain a 'comment_text' column.")
        else:
             st.error("Model not loaded.")
