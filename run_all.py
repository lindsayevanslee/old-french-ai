import subprocess

# List of scripts to run
preprocessing_scripts = ["delete-non-images.py", 
           "convert-heic-to-png.py", 
           "checks.py"]

for script in preprocessing_scripts:
    subprocess.run(["python", script])