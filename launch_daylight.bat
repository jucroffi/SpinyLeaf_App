@echo off
call "C:\SpinyLeaf\WPy64-31190b5\scripts\activate.bat" spinyleaf
streamlit run "C:\SpinyLeaf\06_SpinyLeaf_Daylight.py" --server.port 8506
