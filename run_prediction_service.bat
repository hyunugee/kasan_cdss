@echo off
echo ===================================================
echo KASAN CDSS - Tacrolimus Dose Prediction Service
echo ===================================================
echo.
echo Setting up Python environment and running Streamlit...
echo.

cd tacrolimus-service

echo Installing requirements...
pip install -r requirements.txt
pip install -r streamlit_app/requirements.txt

echo.
echo Starting Streamlit App (Port 8501)...
echo Please keep this window open while using the Prediction tab.
echo.

streamlit run streamlit_app/app.py --server.port 8501 --server.headings.visible false
