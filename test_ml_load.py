import os
import django
from django.conf import settings

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'polling_project.settings')
django.setup()

from verification.ml_utils import FaceVerificationSystem

print("Testing ML System Loading...")
fv = FaceVerificationSystem()
print("ML System Loaded.")
print(f"InsightFace Available: {getattr(fv, 'app', None) is not None}")
print(f"MediaPipe Available: {getattr(fv, 'face_mesh', None) is not None}")
