import face_recognition
import cv2

# Test import van face_recognition_models
try:
    import face_recognition_models
    print("face_recognition_models is correct geïnstalleerd ✅")
except ImportError:
    print("face_recognition_models mist nog ❌")

print(f"face_recognition versie: {face_recognition.__version__}")
print(f"OpenCV versie: {cv2.__version__}")
