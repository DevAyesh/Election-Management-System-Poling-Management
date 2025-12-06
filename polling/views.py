from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import AuthenticationForm
from django.http import JsonResponse
from .db_utils import get_db
import base64
import json
from datetime import datetime
import cv2
import numpy as np

def is_poll_worker(user):
    """Check if user is a poll worker (staff member)"""
    return user.is_staff

@login_required
@user_passes_test(is_poll_worker)
def polling_home(request):
    """Main polling verification page"""
    return render(request, 'polling/home.html')

@login_required
@user_passes_test(is_poll_worker)
def verify_nic(request):
    """Search voter by NIC"""
    if request.method == 'POST':
        data = json.loads(request.body)
        nic = data.get('nic', '').strip()
        
        if not nic:
            return JsonResponse({'error': 'NIC is required'}, status=400)
        
        # Search in voters collection
        db = get_db()
        voter = db.voters.find_one({'nic': nic})
        
        if not voter:
            return JsonResponse({'error': 'Voter not found'}, status=404)
        
        # Check if already verified today
        verification_log = db.verification_logs.find_one({
            'nic': nic,
            'verified': True,
            'date': datetime.now().strftime('%Y-%m-%d')
        })
        
        if verification_log:
            return JsonResponse({
                'error': 'This voter has already been verified today',
                'already_verified': True
            }, status=400)
        
        # Return voter data (excluding _id)
        voter_data = {
            'nic': voter.get('nic'),
            'name': voter.get('name'),
            'date_of_birth': voter.get('date_of_birth'),
            'gender': voter.get('gender'),
            'admin_district': voter.get('admin_district'),
            'election_district': voter.get('election_district'),
            'polling_division': voter.get('polling_division'),
            'face_image': voter.get('face_image')  # Base64 image
        }
        
        return JsonResponse({'voter': voter_data})
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

@login_required
@user_passes_test(is_poll_worker)
def verify_face(request):
    """Verify face using ML (Anti-spoof, Blink, Match)"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            nic = data.get('nic')
            live_image = data.get('live_image')  # Base64 image from webcam
            
            if not nic or not live_image:
                return JsonResponse({'error': 'Missing data'}, status=400)
            
            # Get stored face image from database
            db = get_db()
            voter = db.voters.find_one({'nic': nic})
            
            if not voter or not voter.get('face_image'):
                return JsonResponse({'error': 'No face image on file'}, status=404)
            
            stored_image_b64 = voter.get('face_image')
            
            # Decode images
            def decode_b64(b64_str):
                if ',' in b64_str:
                    b64_str = b64_str.split(',')[1]
                img_bytes = base64.b64decode(b64_str)
                nparr = np.frombuffer(img_bytes, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            live_img = decode_b64(live_image)
            stored_img = decode_b64(stored_image_b64)

            # Initialize ML System
            from verification.ml_utils import FaceVerificationSystem
            fv_system = FaceVerificationSystem()

            # 1. Anti-Spoofing
            is_spoof = fv_system.check_spoof(live_img)
            if is_spoof:
                return JsonResponse({
                    'verified': False,
                    'message': 'Spoof detected! Please use a real face.',
                    'confidence': 0
                })

            # 2. Liveness (Blink)
            # For single-shot verification, we might skip strict blink check or just log it
            has_blinked = fv_system.check_liveness(live_img)
            # If we want to enforce blink, we could fail here, but usually blink is a sequence check.
            # We will proceed but note it.

            # 3. Face Matching
            # Generate embeddings
            live_embedding = fv_system.get_embedding(live_img)
            stored_embedding = fv_system.get_embedding(stored_img)

            if live_embedding is None:
                 return JsonResponse({'verified': False, 'message': 'No face detected in live camera', 'confidence': 0})
            if stored_embedding is None:
                 return JsonResponse({'verified': False, 'message': 'No face detected in stored photo', 'confidence': 0})

            # Compare
            is_match, score = fv_system.verify_face(live_img, stored_embedding) # verify_face calculates live embedding again, optimization possible but this is cleaner API usage if we refactor verify_face to take embedding. 
            # Actually verify_face takes (img_array, registered_embedding). So we pass live_img and stored_embedding.
            
            # Recalculate score for display (0 to 100)
            confidence = round(score * 100, 2)

            # Log verification attempt
            db.verification_logs.insert_one({
                'nic': nic,
                'timestamp': datetime.now(),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'verified': is_match,
                'spoof': is_spoof,
                'blink': has_blinked,
                'confidence': confidence,
                'poll_worker': request.user.username
            })
            
            return JsonResponse({
                'verified': is_match,
                'confidence': confidence,
                'message': 'Face verified successfully!' if is_match else 'Face does not match registered photo'
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({
                'error': f'Face verification error: {str(e)}',
                'verified': False
            }, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

def poll_worker_login(request):
    """Login page for poll workers"""
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            if user.is_staff:
                login(request, user)
                return redirect('polling:home')
            else:
                form.add_error(None, "You are not authorized as a poll worker")
    else:
        form = AuthenticationForm()
    
    return render(request, 'polling/login.html', {'form': form})
