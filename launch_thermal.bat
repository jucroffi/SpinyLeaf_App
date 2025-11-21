@echo off
call "C:\SpinyLeaf\WPy64-31190b5\scripts\activate.bat" spinyleaf
streamlit run "C:\SpinyLeaf\05_SpinyLeaf_Thermal.py" --server.port 8505
