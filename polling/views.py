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
            live_images_b64 = data.get('live_images')
            live_image_b64 = data.get('live_image')
            
            if not nic or (not live_images_b64 and not live_image_b64):
                return JsonResponse({'error': 'Missing data'}, status=400)
            
            if live_images_b64 is None:
                live_images_b64 = [live_image_b64]
            
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

            stored_img = decode_b64(stored_image_b64)

            # Initialize ML System
            from verification.ml_utils import FaceVerificationSystem
            fv_system = FaceVerificationSystem()
            
            stored_embedding, status = fv_system.get_embedding(stored_img, min_face_width=0)
            if status != "OK":
                 return JsonResponse({'verified': False, 'message': 'No valid face detected in stored photo', 'confidence': 0})
            
            best_score = -1.0
            best_match = False
            best_spoof = False
            best_blink = False
            any_face_detected = False
            any_face_too_small = False

            for img_b64 in live_images_b64:
                live_img = decode_b64(img_b64)
                
                live_embedding, status = fv_system.get_embedding(live_img, min_face_width=80)
                if status == "TOO_SMALL":
                    any_face_too_small = True
                    continue
                if status != "OK":
                    continue
                    
                any_face_detected = True
                
                is_spoof = fv_system.check_spoof(live_img)
                has_blinked = fv_system.check_liveness(live_img)
                
                is_match, score = fv_system.compare_embeddings(live_embedding, stored_embedding)
                is_match = bool(is_match)
                score = float(score)
                
                if score > best_score:
                    best_score = score
                    best_match = is_match
                    best_spoof = is_spoof
                    best_blink = has_blinked

            if not any_face_detected:
                if any_face_too_small:
                    return JsonResponse({'verified': False, 'message': 'Face too small. Move closer to the camera.', 'confidence': 0})
                return JsonResponse({'verified': False, 'message': 'No face detected in live camera. Look at the camera and try again.', 'confidence': 0})
            
            if best_spoof:
                return JsonResponse({
                    'verified': False,
                    'message': 'Spoof detected! Please use a real face.',
                    'confidence': 0
                })

            # Recalculate score for display (0 to 100)
            confidence = round(best_score * 100, 2)

            # Log verification attempt
            db.verification_logs.insert_one({
                'nic': nic,
                'timestamp': datetime.now(),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'verified': best_match,
                'spoof': best_spoof,
                'blink': best_blink,
                'confidence': confidence,
                'poll_worker': request.user.username
            })
            
            return JsonResponse({
                'verified': best_match,
                'confidence': confidence,
                'message': 'Face verified successfully!' if best_match else 'Face does not match registered photo.'
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({
                'error': f'Face verification error: {str(e)}',
                'message': 'An error occurred during face verification.',
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
