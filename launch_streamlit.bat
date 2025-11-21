@echo off
call "C:\SpinyLeaf_APP_Dep\WPy64-31190b5\scripts\activate.bat" spinyleaf
streamlit run "C:\SpinyLeaf_APP_Dep\01_SpinyLeaf_Comfort.py" --server.port 8501
