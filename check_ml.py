import sys

print("Python executable:", sys.executable)

try:
    import insightface
    print(f"InsightFace is installed. Version: {insightface.__version__}")
except ImportError:
    print("InsightFace is NOT installed.")

try:
    import deepface
    from deepface import DeepFace
    print(f"DeepFace is installed. Version: {deepface.__version__}")
except ImportError:
    print("DeepFace is NOT installed.")

try:
    import mediapipe
    print(f"MediaPipe is installed. Version: {mediapipe.__version__}")
except ImportError:
    print("MediaPipe is NOT installed.")
