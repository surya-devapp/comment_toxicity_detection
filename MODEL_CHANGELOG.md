# Model Version Changelog

## v1: Baseline (toxicity_model.pkl)
- **Algorithm**: MLPClassifier (Neural Network)
- **Features**: TF-IDF (Top 5000 words)
- **Target**: Binary 'toxic' label from Jigsaw dataset
- **Status**: Deprecated (High False Positives on complaints)

## v2: Calibrated (toxicity_model_calibrated.pkl)
- **Improvement**: Wrapped MLP in `CalibratedClassifierCV` (Sigmoid, 3-fold)
- **Goal**: Fix over-confidence (0.8 score for mild complaints)
- **Status**: Active (Better probability estimates, but still struggles with "I hate you" vs "I hate this")

## v3: Feature Engineered (toxicity_model_v3_enhanced.pkl) - PLANNED
- **Improvement**: Adding Hand-crafted Features
    1. **Target Detection**: Identifying usage of 2nd person pronouns (you, your) to detect personal attacks.
    2. **Sentiment Polarity**: Adding sentiment measure to distinguish strong negative emotion from descriptive text.
    3. **Metadata**: Text length, caps ratio (shouting).
- **Goal**: Distinguish between "Attacking a person" (Toxic) vs "Complaining about a thing" (Safe/Complaint).
