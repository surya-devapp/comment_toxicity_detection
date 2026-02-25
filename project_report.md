# Deep Learning for Comment Toxicity Detection with Streamlit

## 1. Project Overview
**Domain**: Online Community Management and Content Moderation

### Problem Statement
Online communities and social media platforms are integral to modern communication. However, the prevalence of toxic comments (harassment, hate speech, offensive language) poses significant challenges to healthy discourse. There is a pressing need for automated systems capable of detecting and flagging toxic comments in real-time.

### Objective
To develop a deep learning-based toxicity model using Python that analyzes text input and predicts the likelihood of toxicity, assisting moderators in taking appropriate actions (filtering, warning, or review).

## 2. Business Use Cases
- **Social Media Platforms**: Real-time filtering of toxic content.
- **Online Forums**: Efficient moderation of user-generated content.
- **Content Moderation Services**: Enhancing moderation capabilities for third-party clients.
- **Brand Safety**: Ensuring advertisements appear in safe online environments.
- **E-learning Platforms**: Creating safer learning environments for students.
- **News Media**: Moderating user comments on articles and posts.

## 3. Techniques Implemented

### Phase 1: Baseline Development
- **Architecture**: Bidirectional LSTM (Bi-LSTM) with 64 hidden units.
- **Preprocessing**: Custom cleaning (regex-based) and word-level tokenization.
- **Why**: Bi-LSTMs are effective for text because they capture context from both previous and subsequent words in a sequence.

### Phase 2: Sensitivity Optimization (Prioritizing Recall)
- **Weighted Loss Function**: Switched to `BCEWithLogitsLoss` with `pos_weight`.
  - *Why*: The dataset is highly imbalanced (toxic comments are rare). Weights (e.g., 5.0 for threats) force the model to penalize false negatives more heavily.
- **Architecture Upgrade**: Increased hidden units to 128 and added a deeper 3-layer classifier head with Dropout (0.3).
  - *Why*: Deeper networks can learn more complex non-linear relationships, and Dropout prevents overfitting on the small toxic sample.
- **Expanded Vocabulary**: Increased features from 10,000 to 15,000 words.

### Phase 3: Semantic Optimization (Eliminating Keyword Bias)
- **Adversarial Fine-tuning**: Created a "Correction Set" of benign idioms (e.g., "killing it", "dying of laughter") and performed targeted fine-tuning with 5x importance weights.
  - *Why*: Global training often causes models to associate specific words (like "kill") with toxicity regardless of context. Targeted fine-tuning "re-educates" the model on these safe idioms.
- **Threshold Calibration (Binary with Edge Case Flagging)**:
  - **Safe (< 0.5)**
  - **Toxic (>= 0.5)**
  - **Edge Case Flag**: Adds `(Needs Human Review)` if the score is in the **0.4 - 0.7** range.
  - *Why*: Provides clear binary decisions while still highlighting borderline cases that require human judgment.

## 3. Results and Proof of Improvement

### Metric Growth
| Metric | Baseline | Phase 2 (Optimized) | Phase 3 (Semantic) |
| :--- | :--- | :--- | :--- |
| **Toxic Recall** | 0.66 | 0.93 | **0.95** |
| **Weighted F2-Score** | 0.52 | 0.75 | **0.76** |
| **Insult Recall** | 0.61 | 0.88 | **0.91** |

### Proof of Work: Resolving Keyword Bias
The following table shows the model's confidence scores for specific "trigger" phrases before and after Phase 3 Fine-tuning.

| Test Case | Phase 1/2 Score (Toxic) | Phase 3 Score (Safe) | Change |
| :--- | :--- | :--- | :--- |
| "I am dying of laughter!" | 0.77 | **0.05** | ✅ -93% risk reduction |
| "That is a killer outfit." | 0.81 | **0.08** | ✅ -90% risk reduction |
| "I will kill you if..." | 0.89 | **0.71** | ✅ Remained Toxic |

## 5. Phase 4: Model Stabilization (Implemented)
To further refine the system, we implemented heuristic-based safety layers:

- **Heuristic Entity Masking**: Capitalized words not appearing at the beginning of a sentence (potential Names/Titles) are masked as `[ENTITY]`.
  - **Result**: "Moby Dick" toxicity score dropped from **0.93** to **0.07**.
- **Language Detection Middleware**: Integrated `langdetect` to identify the input language.
  - **Result**: Users get a warning if they input non-English text, as the model's vocabulary is English-centric.

## 6. Future Outlook: Multilingual & Advanced NER
While baseline stabilization is complete, future iterations should focus on:
- **Zero-shot Transfer**: Using cross-lingual models like **XLM-RoBERTa** for native support of 100+ languages.
- **Advanced NER**: Using SpaCy or HuggingFace NER models for more precise entity detection than the current heuristic.

## 9. Technical Implementation Deep-Dive

### A. Handling the 6 Toxicity Types (Multi-label)
Unlike standard classification, our model performs **Multi-label Classification**. 
- **The Concept**: A comment can be both an `insult` and `obscene` at the same time. 
- **The Implementation**: The output layer of the model has 6 neurons. We use a **Sigmoid** activation on each neuron, allowing them to independently signal 0 or 1.
- **Decision Logic**: If an input is "Safe," the training process (Binary Cross Entropy) pushes all 6 outputs toward zero.

### B. Eliminating Data Errors (Data Cleaning)
To prevent "ERROR" codes and unreliable scores, we implemented a strict data ingestion pipeline:
1. **NaN Removal**: Using `df.dropna(subset=['comment_text'])` to eliminate missing values.
2. **Whitespace Filtering**: Filtering out comments that are just spaces or empty strings.
3. **Safe Logic**: In `utils.py`, we added a check to return `unknown` if the input is empty, rather than crashing the language detector.

### C. The Training Pipeline
1. **Tokenization**: Converting words into numerical IDs based on a 15,000-word vocabulary.
2. **Bi-LSTM Hidden Layers**: Processing text in two directions to capture context (e.g., distinguishing "killing it" from "threatening to kill").
3. **Weighted Penalty**: Since "Threats" are rare (minority class), we applied a **5.0 weight** to the loss function, forcing the model to learn them with high priority.

## 7. Professional Value: Human-in-the-Loop (HITL)
For project submission, the binary status with a **Needs Human Review** flag represents a sophisticated safety strategy: 
- **Clear Decisiveness**: The system provides a primary "Safe" or "Toxic" label, ensuring clear actionability.
- **Handling Ambiguity**: By appending the review flag to edge cases (0.4 - 0.7), we acknowledge the ambiguity of language (e.g., sarcasm or cultural slang) without stalling the workflow.
- **Business Impact**: This approach balances **Community Safety** with **User Retention**, offering a high-precision moderation tool.

## 8. Evaluation Metrics Decoded
During your live evaluation, you may need to explain these metrics in detail:

### A. Precision (The "Quality" Metric)
- **Concept**: If the model flags 100 comments as toxic, and 90 are actually toxic, your Precision is **90%**.
- **Impact**: High precision means fewer "false alarms." You aren't accidentally tagging safe users as toxic.

### B. Recall (The "Safety" Metric)
- **Concept**: If there are 100 total toxic comments in the dataset, and the model catches 95 of them, your Recall is **95%**.
- **Impact**: This is our most important metric. High recall ensures that threats and hate speech don't slip through the filter.

### C. F1-Score (The Balanced Mean)
- **Concept**: A single number that balances both Precision and Recall equally.
- **When to use**: When you care about false positives and false negatives equally.

### D. F2-Score (The Safety-First Mean)
- **Concept**: A weighted average where **Recall is given twice the importance of Precision**.
- **Why we used it**: In content moderation, missing a dangerous threat (Low Recall) is a "Catastrophic Failure," whereas accidentally flagging a benign idiom (Low Precision) is just a "Minor Inconvenience" because a human can still review it. F2 proves that our model prioritizes community safety.
