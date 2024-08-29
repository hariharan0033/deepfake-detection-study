import subprocess

# Evaluate XceptionNet
subprocess.run(["python", "models/xception/evaluate.py"])

# Evaluate EfficientNet
subprocess.run(["python", "models/efficientnet/evaluate.py"])

# Evaluate Meso4
subprocess.run(["python", "models/meso4/evaluate.py"])
