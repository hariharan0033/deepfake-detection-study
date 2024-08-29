import subprocess

# Train XceptionNet
subprocess.run(["python", "models/xception/train.py"])

# Train EfficientNet
subprocess.run(["python", "models/efficientnet/train.py"])

# Train Meso4
subprocess.run(["python", "models/meso4/train.py"])
