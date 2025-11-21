@echo off
call "C:\SpinyLeaf\WPy64-31190b5\scripts\activate.bat" spinyleaf
streamlit run "C:\SpinyLeaf\03_SpinyLeaf_Social.py" --server.port 8503
