@echo off
call "C:\SpinyLeaf\WPy64-31190b5\scripts\activate.bat" spinyleaf
streamlit run "C:\SpinyLeaf\07_SpinyLeaf_AirQuali.py" --server.port 8507
