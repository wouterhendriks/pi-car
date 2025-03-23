import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

try:
    import face_recognition_models
    print(f"face_recognition_models found at: {face_recognition_models.__path__}")

    # List the contents of the models directory
    import os
    models_dir = face_recognition_models.__path__[0]
    print(f"Contents of {models_dir}:")
    for item in os.listdir(models_dir):
        print(f"  {item}")

except ImportError:
    print("face_recognition_models not found in Python path")
except Exception as e:
    print(f"Error importing face_recognition_models: {e}")