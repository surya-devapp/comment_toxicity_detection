import streamlit as st
import pandas as pd
import torch
from utils import load_inference_model, predict_toxicity, get_toxicity_status, detect_language
import plotly.express as px
import os

st.set_page_config(page_title="Deep Learning for Comment Toxicity Detection with Streamlit", layout="wide")

st.title("🛡️ Deep Learning for Comment Toxicity Detection with Streamlit")
st.markdown("""
### Online Community Management and Content Moderation
This application uses a Advanced Deep Learning (Bi-LSTM) model to detect toxic content in real-time. 
By accurately identifying toxicity, it assists administrators in maintaining healthy online discourse.
""")

# Load model and tokenizer
@st.cache_resource
def get_model(model_mtime):
    model_path = 'models/toxicity_model.pth'
    tokenizer_path = 'models/tokenizer.pkl'
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        return load_inference_model(model_path, tokenizer_path, vocab_size=15000)
    return None, None

# Get model modification time to use as a cache breaker
model_file = 'models/toxicity_model.pth'
mtime = os.path.getmtime(model_file) if os.path.exists(model_file) else 0
model, tokenizer = get_model(mtime)

if model is None:
    st.warning("Model and tokenizer not found. Please run the training script first.")
    st.stop()

# Sidebar
st.sidebar.title("Model Calibration")
st.sidebar.info("Adjust thresholds to balance sensitivity (Recall) vs False Positives (Precision).")
safe_t = st.sidebar.slider("Safe Threshold (Max score below this is Safe)", 0.0, 1.0, 0.4)
toxic_t = st.sidebar.slider("Toxic Threshold (Scores above this are Toxic)", 0.0, 1.0, 0.7)

st.sidebar.title("Model Stability")
st.sidebar.info("Stabilization techniques to reduce false positives.")
mask_entities = st.sidebar.checkbox("Mask Proper Nouns (NER Heuristic)", value=True, help="Replaces capitalized words (e.g., names, titles) with [ENTITY] to avoid keyword triggers.")

# Tabs
tab1, tab2, tab3 = st.tabs(["Real-time Prediction", "Bulk Prediction", "Model Performance & Insights"])

with tab1:
    st.subheader("Analyze a Comment")
    comment = st.text_area("Enter your comment here:", placeholder="Type something...")
    
    if st.button("Predict"):
        if comment:
            # Language Check
            lang = detect_language(comment)
            if lang != 'en':
                st.warning(f"🌐 **Language Alert**: Detected language is '{lang}'. This model is optimized for English and may produce unreliable results in other languages.")
            
            results, cleaned_text = predict_toxicity(comment, model, tokenizer, mask_entities=mask_entities)
            status, color = get_toxicity_status(results, safe_t, toxic_t)
            
            # Entity Masking Warning
            if '[ENTITY]' in cleaned_text:
                st.info("ℹ️ **Sensitivity Note**: Proper nouns were detected and masked to prevent false positives.")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"### Status: :{color}[{status}]")
                # Show cleaning details
                with st.expander("🔍 See Text Cleaning Details"):
                    st.write(f"**Original:** {comment}")
                    st.write(f"**Cleaned/Masked:** {cleaned_text}")
                    st.caption("Note: Cleaning removes noise and masks proper nouns to prevent false positives.")
                
                for label, prob in results.items():
                    st.markdown(f"**{label.capitalize()}**: {prob:.4f}")
                    st.progress(float(prob))
            
            with col2:
                # Plot results
                df_res = pd.DataFrame(list(results.items()), columns=['Category', 'Probability'])
                fig = px.bar(df_res, x='Category', y='Probability', 
                             color='Probability', color_continuous_scale='Reds',
                             title="Toxicity Levels Visualization")
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
                
            # Final Message
            if "Toxic" in status:
                if "(Needs Human Review)" in status:
                    st.warning("🧐 **Edge Case**: This comment is borderline but leans towards toxicity. Human review is recommended.")
                else:
                    st.error("⚠️ This comment contains highly toxic content!")
            else:
                if "(Needs Human Review)" in status:
                    st.info("ℹ️ **Edge Case**: This comment is mostly safe but contains potentially ambiguous language.")
                else:
                    st.success("✅ This comment appears to be safe.")
        else:
            st.warning("Please enter some text first.")

with tab2:
    st.subheader("Bulk Prediction (CSV)")
    uploaded_file = st.file_uploader("Upload a CSV file containing comments", type=["csv"])
    
    if uploaded_file is not None:
        df_bulk = pd.read_csv(uploaded_file)
        if 'comment_text' in df_bulk.columns:
            if st.button("Run Bulk Analysis"):
                with st.spinner("Analyzing..."):
                    bulk_results = []
                    for comment in df_bulk['comment_text']:
                        comment_str = str(comment).strip()
                        if not comment_str or "#ERROR!" in comment_str:
                            res = {c: 0.0 for c in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']}
                            status = "Error/Empty"
                            cleaned = "N/A"
                        else:
                            res, cleaned = predict_toxicity(comment_str, model, tokenizer, mask_entities=mask_entities)
                            status, color = get_toxicity_status(res, safe_t, toxic_t)
                        
                        res['status'] = status
                        res['cleaned_text'] = cleaned
                        bulk_results.append(res)
                    
                    res_df = pd.DataFrame(bulk_results)
                    final_df = pd.concat([df_bulk, res_df], axis=1)
                    
                    # Reorder to show Original vs Cleaned side-by-side
                    cols = ['comment_text', 'cleaned_text', 'status'] + [c for c in res_df.columns if c not in ['status', 'cleaned_text']]
                    final_df = final_df[cols]
                    
                    st.write("### Prediction Results (Cleaned Data Comparison)")
                    st.dataframe(final_df, use_container_width=True)
                    
                    csv = final_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Results", csv, "toxicity_results.csv", "text/csv")
        else:
            st.error("CSV must contain a 'comment_text' column.")

with tab3:
    st.header("📊 Model Metrics & Insights")
    
    col_m1, col_m2 = st.columns(2)
    col_m3, col_m4 = st.columns(2)
    
    with col_m1:
        st.metric("Recall (Safety)", "0.95", "Toxic Class Coverage")
    with col_m2:
        st.metric("Precision (Accuracy)", "0.78", "Weighted Average")
    with col_m3:
        st.metric("Weighted F1-Score", "0.67", "Balanced Mean")
    with col_m4:
        st.metric("Weighted F2-Score", "0.76", "Recall-Weighted Mean")
    
    st.markdown("---")
    
    st.subheader("💡 Sample Test Cases")
    st.info("Copy and paste these into the Real-time Predictor to see the model in action.")
    
    samples = [
        {"Comment": "I am dying of laughter!", "Expected": "Safe (Idiom)", "Feature": "Phase 3: Semantic Optimization"},
        {"Comment": "Moby Dick by Herman Melville.", "Expected": "Safe (Proper Noun)", "Feature": "Phase 4: Entity Masking"},
        {"Comment": "You are a complete idiot and I hate you.", "Expected": "Toxic", "Feature": "Phase 2: Sensitivity"},
        {"Comment": "This is a killer outfit you're wearing!", "Expected": "Safe (Idiom)", "Feature": "Phase 3: Semantic Optimization"},
        {"Comment": "I will kill you if you ever come back here.", "Expected": "Toxic (Threat)", "Feature": "Phase 2: Sensitivity"}
    ]
    st.table(samples)

    st.markdown("---")
    st.subheader("🛠️ Technical Specifications")
    st.markdown("""
    - **Architecture**: 3-Layer Bidirectional LSTM with Dropout (0.3).
    - **Optimization**: Weighted Cross-Entropy Loss (5.0 weight on toxic classes).
    - **Stability**: Heuristic Named Entity Recognition (NER) for Proper Noun masking.
    - **Dataset**: Jigsaw/Wikipedia Talk Page edits (150,000+ samples).
    """)

st.sidebar.markdown("---")
st.sidebar.info("Built with PyTorch & Streamlit")
