@echo off
call "C:\SpinyLeaf\WPy64-31190b5\scripts\activate.bat" spinyleaf
streamlit run "C:\SpinyLeaf\01_SpinyLeaf_Comfort.py" --server.port 8501
