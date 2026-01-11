@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo Training model...
python src/train.py

echo Starting Streamlit App...
streamlit run app.py
pause
