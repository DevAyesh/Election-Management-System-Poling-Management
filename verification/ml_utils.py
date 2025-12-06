import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try importing InsightFace
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

# Try importing DeepFace
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

# Try importing MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

class FaceVerificationSystem:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaceVerificationSystem, cls).__new__(cls)
            cls._instance.initialize_models()
        return cls._instance

    def initialize_models(self):
        print(f"Initializing models... InsightFace: {INSIGHTFACE_AVAILABLE}, DeepFace: {DEEPFACE_AVAILABLE}, MediaPipe: {MEDIAPIPE_AVAILABLE}")
        
        if INSIGHTFACE_AVAILABLE:
            try:
                # providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if GPU available
                self.app = FaceAnalysis(name='buffalo_l') 
                self.app.prepare(ctx_id=0, det_size=(640, 640))
                print("InsightFace initialized.")
            except Exception as e:
                print(f"Failed to init InsightFace: {e}")
                self.app = None
        
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("MediaPipe initialized.")

    def get_embedding(self, img_array):
        if INSIGHTFACE_AVAILABLE and getattr(self, 'app', None):
            faces = self.app.get(img_array)
            if not faces:
                return None
            largest_face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
            return largest_face.embedding
        
        if DEEPFACE_AVAILABLE:
            try:
                # DeepFace.represent returns a list of dicts
                embedding_objs = DeepFace.represent(img_array, model_name="ArcFace", enforce_detection=False)
                if embedding_objs:
                    return np.array(embedding_objs[0]["embedding"])
            except Exception as e:
                print(f"DeepFace embedding error: {e}")
        
        return None

    def verify_face(self, img_array, registered_embedding, threshold=0.35):
        live_embedding = self.get_embedding(img_array)
        if live_embedding is None:
            return False, 0.0
        
        # Cosine Similarity
        # Ensure shapes match
        if registered_embedding.shape != live_embedding.shape:
             # Handle dimension mismatch if switching models (e.g. 512 vs 128)
             # For now assume same model used for registration and verification
             pass

        sim = np.dot(registered_embedding, live_embedding) / (np.linalg.norm(registered_embedding) * np.linalg.norm(live_embedding))
        
        # Threshold might need adjustment based on model (ArcFace usually 0.5 is good, DeepFace default is different)
        # For ArcFace, cosine distance threshold is ~0.68, so similarity threshold is ~0.32
        print(f"DEBUG: Similarity Score: {sim}, Threshold: {threshold}")
        return sim > threshold, sim

    def check_liveness(self, img_array):
        if MEDIAPIPE_AVAILABLE and hasattr(self, 'face_mesh'):
            # Calculate EAR
            results = self.face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Simple logic: just return True if face found for now, 
                    # real EAR calculation requires mapping landmarks
                    return True 
            return False
        
        # Fallback: Face detection implies "liveness" in simplest form (not secure)
        # Or use OpenCV eye detector
        return True

    def check_spoof(self, img_array):
        # Placeholder for Anti-Spoofing
        # Real implementation needs MiniFASNet
        return False

