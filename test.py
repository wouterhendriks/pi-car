import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

try:
    import face_recognition
    print(f"✅face_recognition version: {face_recognition.__version__}")
except Exception as e:
    print(f"Error importing face_recognition: {e}")

    try:
        import face_recognition_models
        print(f"face_recognition_models found at: {face_recognition_models.__path__}")
    except ImportError:
        print("face_recognition_models not found in Python path")
    except Exception as e:
        print(f"Error importing face_recognition_models: {e}")
