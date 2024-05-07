import subprocess

def start_script(path):
    subprocess.Popen(["python", path])

start_script("app.py")
start_script("column_extractor_api.py")
