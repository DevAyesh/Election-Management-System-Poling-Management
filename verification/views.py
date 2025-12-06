import base64
import cv2
import numpy as np
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Voter
from .ml_utils import FaceVerificationSystem

@csrf_exempt
def register_voter(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            slin_nic = data.get('slin_nic')
            name = data.get('name')
            image_data = data.get('image') # Base64 string

            if not slin_nic or not image_data:
                return JsonResponse({'error': 'Missing NIC or Image'}, status=400)

            # Decode image
            try:
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                img_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                return JsonResponse({'error': f'Invalid image data: {str(e)}'}, status=400)

            # Generate Embedding
            fv_system = FaceVerificationSystem()
            embedding = fv_system.get_embedding(img)

            if embedding is None:
                return JsonResponse({'error': 'No face detected in image'}, status=400)

            # Save to MongoDB
            # Convert numpy array to list for JSON serialization in MongoDB
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            
            voter = Voter(slin_nic=slin_nic, name=name, face_embedding=embedding_list)
            voter.save()

            return JsonResponse({'message': 'Voter registered successfully', 'slin_nic': slin_nic})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@csrf_exempt
def verify_voter(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            slin_nic = data.get('slin_nic')
            image_data = data.get('image')

            if not slin_nic or not image_data:
                return JsonResponse({'error': 'Missing NIC or Image'}, status=400)

            # Decode image
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            fv_system = FaceVerificationSystem()

            # 1. Anti-Spoofing
            is_spoof = fv_system.check_spoof(img)
            if is_spoof:
                 return JsonResponse({
                    'verified': False,
                    'spoof': True,
                    'blink': False,
                    'match': False,
                    'message': 'Spoof detected'
                })

            # 2. Liveness (Blink) - This is usually accumulated on frontend, 
            # but here we return current frame status
            has_blinked = fv_system.check_liveness(img)

            # 3. Face Matching
            voter = Voter.get_by_nic(slin_nic)
            if not voter:
                return JsonResponse({'error': 'Voter not found'}, status=404)

            is_match, score = fv_system.verify_face(img, np.array(voter.face_embedding))

            verified = (not is_spoof) and is_match # Blink is usually a session state, ignored for single frame strict verification unless passed

            return JsonResponse({
                'verified': verified,
                'spoof': is_spoof,
                'blink': has_blinked,
                'match': is_match,
                'score': float(score)
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Method not allowed'}, status=405)

def verification_page(request):
    from django.shortcuts import render
    return render(request, 'verification/verify.html')

