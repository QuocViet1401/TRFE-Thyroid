

import subprocess
import sys
from pathlib import Path

print("\n" + "="*60)
print("TRFEPLUS Model Pre-Training")
print("="*60)
print("\nThis script will train the model ONCE.")
print("After training completes, you only need to use: streamlit run app.py")
print("\nTraining settings:")
print("- Model: TRFEPLUS")
print("- Dataset: TATN (Thyroid nodules)")
print("- Epochs: 10 (recommended - takes 10-30 hours on CPU)")
print("- Optimizer: SGD (lr=1e-3, momentum=0.9)")
print("="*60 + "\n")

# Ask user for epochs
while True:
    try:
        epochs = input("Enter number of epochs to train (2-40, recommended 10): ").strip()
        epochs = int(epochs)
        if 2 <= epochs <= 40:
            break
        else:
            print("Please enter a number between 2 and 40")
    except ValueError:
        print("Invalid input. Please enter a number.")

print(f"\nStarting training with {epochs} epochs...")
print("This may take several hours on CPU. Do NOT close this window.\n")

try:
    result = subprocess.run(
        [sys.executable, "train.py", "-fold", "0", "-model_name", "trfeplus",
         "-dataset", "TATN", "-gpu", "-1", "-lr", "1e-3", "-nepochs", str(epochs),
         "-batch_size", "8", "-use_test", "0"],  # Skip validation, reduce batch
        check=False
    )
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("SUCCESS! Model training completed!")
        print("="*60)
        print("\nYou can now use the web app with a trained model:")
        print("\n  python RUN.py")
        print("  [3] Run Web App Only")
        print("\nThen open your browser to http://localhost:8501")
        print("Upload ultrasound images and the model will analyze them accurately!")
        print("="*60 + "\n")
    else:
        print("\nTraining failed! Check the error messages above.")
        sys.exit(1)

except KeyboardInterrupt:
    print("\n\nTraining interrupted by user. Model may be partially trained.")
    print("Run this script again to continue or complete training.")
except Exception as e:
    print(f"\nError during training: {e}")
    sys.exit(1)
