import subprocess

# List of scripts to run
preprocessing_scripts = ["delete-non-images.py", 
           "convert-heic-to-png.py", 
           "checks.py",
           "reduce-size.py",
           "remove-background.py",
           "correct-skew.py"]

for script in preprocessing_scripts:
    subprocess.run(["python", script])