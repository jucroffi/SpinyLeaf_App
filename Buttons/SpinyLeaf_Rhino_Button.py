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

    # Create directories if they don't exist
    if not os.path.exists(main_f):
        os.mkdir(main_f)
    if not os.path.exists(model_f):
        os.mkdir(model_f)

    # Save a copy to the destination
    Rhino.RhinoDoc.ActiveDoc.WriteFile(dest_path, Rhino.FileIO.FileWriteOptions())

    app_path = r"C:\Users\jucro\Box\PhD\Python_scripts\SpinyLeaf_Tests\00_App_streamlit_rhino_test_01.py"
    #bat_path = r"C:\Users\jucro\Box\PhD\Python_scripts\SpinyLeaf_Tests\launch_streamlit.bat"
    bat_path = "../python/python.exe streamlit run \"" + app_path + "\" --server.port 8508\n"

    #with open(bat_path, "w") as f:
    #    f.write("@echo off\n")
    #    f.write("call C:\\Users\\jucro\\anaconda3\\Scripts\\activate.bat SpinyLeaf_Python311\n")
    #    f.write("streamlit run \"" + app_path + "\"\n")

    subprocess.Popen(["explorer.exe", bat_path])

save_and_run_streamlit()