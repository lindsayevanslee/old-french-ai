import subprocess
import os

#find the path of the current script
current_script_path = os.path.dirname(os.path.abspath(__file__))
print(f"Current script path: {current_script_path}")

# List of scripts to run
preprocessing_scripts = ["delete-non-images.py", 
           "convert-heic-to-png.py", 
           "checks.py",
           "reduce-size.py",
           "remove-background.py",
           "correct-skew.py"]

for script in preprocessing_scripts:

    # Get the full path of the script
    script_path = os.path.join(current_script_path, script)
    print(f"Running script: {script_path}")

    # Run the script
    subprocess.run(["python", script_path])