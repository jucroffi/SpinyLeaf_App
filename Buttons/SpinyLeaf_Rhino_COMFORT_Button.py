import os
import subprocess
import Rhino
import scriptcontext as sc

try:
    from pathlib import Path
except ImportError:
    import os
    class Path:
        @staticmethod
        def home():
            return os.path.expanduser("~")

def save_and_run_streamlit():
    current_path = Rhino.RhinoDoc.ActiveDoc.Path

    if not current_path:
        print("The current Rhino file has never been saved. Please save it first.")
        return

    home_path = os.path.expanduser("~")
    main_f = os.path.join(home_path, "SpinyLeaf_App")
    model_f = os.path.join(main_f, "Rhino_Model")
    dest_path = os.path.join(model_f, "shared_model.3dm")

    if not os.path.exists(main_f):
        os.mkdir(main_f)
    if not os.path.exists(model_f):
        os.mkdir(model_f)

    Rhino.RhinoDoc.ActiveDoc.WriteFile(dest_path, Rhino.FileIO.FileWriteOptions())


    app_path = r"C:\SpinyLeaf\01_SpinyLeaf_Comfort.py"
    bat_path = r"C:\SpinyLeaf\launch_comfort.bat"

    with open(bat_path, "w") as fh:
        fh.write("@echo off\r\n")
        fh.write('call "C:\\SpinyLeaf\\WPy64-31190b5\\scripts\\activate.bat" spinyleaf\r\n')
        fh.write('streamlit run "{}" --server.port 8501\r\n'.format(app_path))

    subprocess.Popen(["cmd.exe", "/k", bat_path])
    

save_and_run_streamlit()