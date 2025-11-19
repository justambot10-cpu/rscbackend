from flask import Flask, request, jsonify, send_from_directory, send_file, redirect, url_for, session
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info and warning messages
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import cv2
import numpy as np
import time
import uuid
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
import json
from dotenv import load_dotenv
import base64
from io import BytesIO
import tempfile
import requests
from deepface import DeepFace

# Load environment variables FIRST
load_dotenv()

app = Flask(__name__, static_folder='public', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'fallback-secret-key-12345')

# ===== ENVIRONMENT VALIDATION =====
def validate_environment():
    """Validate that all required environment variables are set"""
    required_vars = ['SUPABASE_URL', 'SUPABASE_KEY', 'FLASK_SECRET_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("   Please check your .env file")
        return False
    
    # Check if using default secret (security warning)
    if os.getenv('FLASK_SECRET_KEY') == 'fallback-secret-key-12345':
        print("❌ WARNING: Using default Flask secret key - this is insecure for production!")
    
    return True

# Validate environment before initialization
if not validate_environment():
    print("❌ Server cannot start due to missing configuration")

# ===== IMPROVED SUPABASE INITIALIZATION =====
supabase = None
supabase_face_recognition = None

def initialize_supabase():
    """Initialize Supabase with better error handling"""
    global supabase, supabase_face_recognition
    
    try:
        SUPABASE_URL = os.getenv('SUPABASE_URL')
        SUPABASE_KEY = os.getenv('SUPABASE_KEY')
        
        print(f"🔄 Supabase URL: {SUPABASE_URL}")
        print(f"🔄 Supabase Key length: {len(SUPABASE_KEY) if SUPABASE_KEY else 0}")
        
        if not SUPABASE_URL or not SUPABASE_KEY:
            print("❌ Missing Supabase credentials")
            return False
        
        from supabase import create_client
        
        # Test the connection
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase client created")
        
        # Try a simple storage operation to test
        try:
            # List buckets to test connection
            buckets = supabase.storage.list_buckets()
            print(f"✅ Supabase connection test passed - {len(buckets)} buckets available")
            
            # Specifically test the readytoservethecommunity bucket
            try:
                files = supabase.storage.from_('readytoservethecommunity').list('training-images')
                file_count = len(files) if files else 0
                print(f"✅ Storage bucket access successful - Found {file_count} files in training-images")
            except Exception as bucket_error:
                print(f"⚠️ Could not access training-images folder: {bucket_error}")
                # This might be normal if the folder doesn't exist yet
                
        except Exception as e:
            print(f"⚠️ Supabase storage test failed: {e}")
            # Try a different test - table query
            try:
                result = supabase.table('_nonexistent_table').select('*').limit(0).execute()
                print("✅ Supabase connection test passed via table query")
            except Exception as e2:
                print(f"❌ Supabase connection failed completely: {e2}")
                return False
        
        # Initialize face recognition
        supabase_face_recognition = SupabaseFaceRecognition(supabase)
        print("✅ Supabase face recognition initialized")
        return True
        
    except ImportError:
        print("❌ Supabase Python client not installed. Run: pip install supabase")
        return False
    except Exception as e:
        print(f"❌ Supabase initialization failed: {e}")
        return False

# Initialize Supabase immediately
if not initialize_supabase():
    print("❌ CRITICAL: Supabase initialization failed. Face recognition will not work.")
else:
    print("✅ Supabase initialized successfully")

# Initialize Firebase (Realtime Database only)
firebase_cred_path = os.getenv('FIREBASE_CREDENTIALS_PATH')
if not firebase_cred_path:
    firebase_cred_path = 'public/assets/json/rscs-1822d-firebase-adminsdk-fbsvc-c2ced89873.json'

if firebase_cred_path and os.path.exists(firebase_cred_path):
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(firebase_cred_path)
            firebase_admin.initialize_app(cred, {
                'databaseURL': os.getenv('FIREBASE_DATABASE_URL', 'https://rscs-1822d-default-rtdb.firebaseio.com/')
            })
        print("✅ Firebase Realtime Database initialized successfully")
    except Exception as e:
        print(f"⚠️ Firebase initialization failed: {e}")
        firebase_db = None
else:
    print("⚠️ Firebase credentials not found. User management will be disabled.")
    firebase_db = None

# Global variables for continuous recognition
continuous_recognition_active = False
frame_count = 0

# Enhanced SupabaseFaceRecognition class with DeepFace recognition
class SupabaseFaceRecognition:
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.known_faces_cache = {}
        self.known_face_embeddings_cache = {}
        self.known_face_roles_cache = {}
        self.known_face_user_ids_cache = {}
        self.known_face_status_cache = {}
        self.last_cache_update = 0
        self.cache_ttl = 300
        print("✅ Enhanced Supabase face recognition with DeepFace initialized")
    
    def get_face_embedding_from_url(self, image_url):
        """Download image from Supabase URL and extract face embedding using DeepFace"""
        try:
            response = requests.get(image_url, timeout=10)
            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    temp_file.write(response.content)
                    temp_file_path = temp_file.name
                
                try:
                    embedding_objs = DeepFace.represent(
                        img_path=temp_file_path,
                        model_name='Facenet',
                        detector_backend='opencv',
                        enforce_detection=False
                    )
                    
                    if embedding_objs and len(embedding_objs) > 0:
                        embedding = embedding_objs[0]['embedding']
                        print(f"✅ Extracted face embedding from {image_url}")
                        return np.array(embedding)
                    else:
                        print(f"⚠️ No face found in {image_url}")
                        return None
                        
                finally:
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                        
            else:
                print(f"❌ Failed to download {image_url}: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error processing image from {image_url}: {e}")
            return None
    
    def refresh_face_cache(self):
        """Force refresh the cache of known faces from Supabase with face embeddings and user roles"""
        if not self.supabase:
            print("❌ Supabase not available for cache refresh")
            return
            
        current_time = time.time()
        if current_time - self.last_cache_update < self.cache_ttl and self.known_face_embeddings_cache:
            return
        
        try:
            print("🔄 FORCE REFRESHING face cache from Supabase with DeepFace...")
            
            # Clear existing caches completely
            self.known_faces_cache = {}
            self.known_face_embeddings_cache = {}
            self.known_face_roles_cache = {}
            self.known_face_user_ids_cache = {}
            self.known_face_status_cache = {}
            
            # List all files in the training-images folder
            response = self.supabase.storage.from_('readytoservethecommunity').list('training-images')
            
            files_data = response
            if hasattr(response, 'data'):
                files_data = response.data
            elif isinstance(response, list):
                files_data = response
            else:
                files_data = []
            
            if not files_data:
                print("📭 No files found in Supabase storage")
                return
                
            processed_files = 0
            successful_embeddings = 0
            
            for file_info in files_data:
                filename = file_info['name']
                
                if not filename or not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                person_name = self.extract_person_name(filename)
                
                if person_name:
                    if person_name not in self.known_faces_cache:
                        self.known_faces_cache[person_name] = []
                        self.known_face_embeddings_cache[person_name] = []
                    
                    public_url = f"https://ptgxqezqawepjaxbqgsx.supabase.co/storage/v1/object/public/readytoservethecommunity/training-images/{filename}"
                    
                    self.known_faces_cache[person_name].append({
                        'filename': filename,
                        'url': public_url,
                        'full_path': f"training-images/{filename}"
                    })
                    
                    print(f"🔍 Extracting face embedding for {person_name} from {filename}")
                    face_embedding = self.get_face_embedding_from_url(public_url)
                    
                    if face_embedding is not None:
                        self.known_face_embeddings_cache[person_name].append(face_embedding)
                        successful_embeddings += 1
                        print(f"   ✅ Face embedding extracted successfully for {person_name}")
                    else:
                        print(f"   ❌ Failed to extract face embedding for {person_name}")
                    
                    processed_files += 1
                    print(f"   ✅ {person_name} -> {filename}")
                    
                    # Get user role and ID from Firebase
                    user_info = self.get_user_info_from_firebase(person_name)
                    self.known_face_roles_cache[person_name] = user_info['role']
                    self.known_face_user_ids_cache[person_name] = user_info['user_id']
                    self.known_face_status_cache[person_name] = user_info['status']
                    print(f"   👤 Role for {person_name}: {user_info['role']}, User ID: {user_info['user_id']}, Status: {user_info['status']}")
                    
                else:
                    print(f"   ❌ Could not extract name from: {filename}")
            
            self.last_cache_update = current_time
            print(f"✅ Face cache refreshed: {len(self.known_faces_cache)} people, {processed_files} images, {successful_embeddings} face embeddings")
            print(f"📝 People in cache: {list(self.known_faces_cache.keys())}")
            print(f"🎭 Roles in cache: {self.known_face_roles_cache}")
                
        except Exception as e:
            print(f"❌ Error refreshing face cache: {e}")
    
    def get_user_info_from_firebase(self, person_name):
        """Get user role and ID from Firebase Realtime Database by matching fullname"""
        try:
            print(f"🔍 Searching Firebase for user: {person_name}")
            
            # Search through all user categories in your actual Firebase structure
            user_categories = ['barangay_captain', 'barangay_official', 'response_team', 'admin']
            
            for category in user_categories:
                ref = db.reference(f'/users/{category}')
                category_users = ref.get()
                
                if category_users:
                    for user_id, user_data in category_users.items():
                        # Check if fullname matches (case insensitive)
                        user_fullname = user_data.get('fullname', '')
                        
                        if user_fullname.lower() == person_name.lower():
                            # Map category to role value
                            role_mapping = {
                                'barangay_captain': 'captain',
                                'barangay_official': 'official', 
                                'response_team': 'response_team',
                                'admin': 'admin'
                            }
                            role_value = role_mapping.get(category, 'unknown')
                            
                            print(f"✅ Found user {person_name}: Category={category}, Role={role_value}, UserID={user_id}")
                            return {
                                'role': role_value,
                                'user_id': user_id,
                                'role_category': category,
                                'fullname': user_fullname,
                                'username': user_data.get('username', ''),
                                'status': user_data.get('status', 'approved')
                            }
            
            print(f"⚠️ No user found for {person_name} in Firebase")
            return {'role': 'unknown', 'user_id': 'unknown', 'role_category': 'unknown', 'status': 'unknown'}
            
        except Exception as e:
            print(f"❌ Error getting user info for {person_name}: {e}")
            return {'role': 'unknown', 'user_id': 'unknown', 'role_category': 'unknown', 'status': 'unknown'}
    
    def extract_person_name(self, filename):
        """Enhanced person name extraction from filename"""
        try:
            clean_filename = filename.split('/')[-1]
            name_without_ext = clean_filename.rsplit('.', 1)[0]
            parts = name_without_ext.split('_')
            
            # Handle timestamp prefix (e.g., "1704234567890_John_Doe_1")
            if len(parts) >= 3 and parts[0].isdigit() and len(parts[0]) >= 10:
                name_parts = parts[1:-1]
                return ' '.join(name_parts) if name_parts else None
            
            # Handle regular format (e.g., "John_Doe_1")
            elif len(parts) >= 2:
                name_parts = parts[:-1]
                return ' '.join(name_parts) if name_parts else None
            
            # Handle simple format (e.g., "John_1")
            elif len(parts) == 2 and not parts[0].isdigit():
                return parts[0]
            
            # If only one part, use it as name
            elif len(parts) == 1 and not parts[0].isdigit():
                return parts[0]
            
            return None
            
        except Exception as e:
            print(f"Error parsing filename {filename}: {e}")
            return None
    
    def recognize_face(self, image_array):
        """REAL face recognition comparing with Supabase-stored faces using DeepFace"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                cv2.imwrite(temp_file.name, image_array)
                temp_file_path = temp_file.name
            
            try:
                embedding_objs = DeepFace.represent(
                    img_path=temp_file_path,
                    model_name='Facenet',
                    detector_backend='opencv',
                    enforce_detection=False
                )
                
                print(f"🔍 Found {len(embedding_objs)} faces in frame")
                
                results = []
                
                for embedding_obj in embedding_objs:
                    face_embedding = np.array(embedding_obj['embedding'])
                    facial_area = embedding_obj['facial_area']
                    
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    left, top, right, bottom = x, y, x + w, y + h
                    
                    best_match_name = "Unknown"
                    best_match_confidence = 0
                    best_match_role = "unknown"
                    best_match_user_id = "unknown"
                    best_match_status = "unknown"
                    best_face_id = None
                    
                    for person_name, person_embeddings in self.known_face_embeddings_cache.items():
                        if not person_embeddings:
                            continue
                            
                        try:
                            for known_embedding in person_embeddings:
                                similarity = self.cosine_similarity(face_embedding, known_embedding)
                                confidence = similarity * 100
                                
                                if confidence > best_match_confidence and confidence > 60:
                                    best_match_name = person_name
                                    best_match_confidence = confidence
                                    best_match_role = self.known_face_roles_cache.get(person_name, 'unknown')
                                    best_match_user_id = self.known_face_user_ids_cache.get(person_name, 'unknown')
                                    best_match_status = self.known_face_status_cache.get(person_name, 'unknown')
                                    best_face_id = f"{person_name}_{hash(person_name)}"
                                    
                                    print(f"✅ Match found: {person_name} ({confidence:.1f}%) - Role: {best_match_role}, Status: {best_match_status}")
                        
                        except Exception as e:
                            print(f"❌ Error comparing with {person_name}: {e}")
                            continue
                    
                    liveness_score = self.detect_liveness(image_array, x, y, w, h)
                    is_live = liveness_score > 0.6
                    
                    if best_match_confidence > 60:
                        results.append({
                            'name': best_match_name,
                            'user_id': best_match_user_id,
                            'confidence': round(best_match_confidence, 2),
                            'role': best_match_role,
                            'status': best_match_status,
                            'liveness_score': float(round(liveness_score, 2)),
                            'is_live': bool(is_live),
                            'face_id': best_face_id or f"face_{len(results)}",
                            'location': {
                                'top': int(top),
                                'right': int(right),
                                'bottom': int(bottom),
                                'left': int(left)
                            },
                            'recognition_type': 'supabase_deepface'
                        })
                    else:
                        results.append({
                            'name': "Unknown",
                            'user_id': 'unknown',
                            'confidence': round(best_match_confidence, 2) if best_match_confidence > 0 else 0,
                            'role': 'unknown',
                            'status': 'unknown',
                            'liveness_score': float(round(liveness_score, 2)),
                            'is_live': bool(is_live),
                            'face_id': f"unknown_{len(results)}",
                            'location': {
                                'top': int(top),
                                'right': int(right),
                                'bottom': int(bottom),
                                'left': int(left)
                            },
                            'recognition_type': 'supabase_deepface'
                        })
                
                print(f"📊 Recognition results: {len(results)} faces")
                return results
                
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            
        except Exception as e:
            print(f"❌ Face recognition error: {e}")
            return []
    
    def cosine_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        try:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0
                
            similarity = dot_product / (norm1 * norm2)
            return max(0, min(1, similarity))
            
        except Exception as e:
            print(f"❌ Error calculating cosine similarity: {e}")
            return 0
    
    def detect_liveness(self, image_array, x, y, w, h):
        """Enhanced liveness detection"""
        try:
            face_roi = image_array[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                return 0.5
            
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            liveness_score = 0.5
            
            variance = np.var(gray_face)
            if variance > 100:
                liveness_score += 0.2
            
            face_area = w * h
            if 5000 < face_area < 50000:
                liveness_score += 0.2
            
            brightness = np.mean(gray_face)
            if 50 < brightness < 200:
                liveness_score += 0.1
            
            edges = cv2.Canny(gray_face, 100, 200)
            edge_density = np.sum(edges > 0) / (w * h)
            if edge_density > 0.05:
                liveness_score += 0.1
            
            return min(1.0, liveness_score)
            
        except Exception as e:
            print(f"Liveness detection error: {e}")
            return 0.5

# Initialize the Supabase face recognition AFTER Supabase is initialized
if supabase:
    supabase_face_recognition = SupabaseFaceRecognition(supabase)
    print("✅ Supabase face recognition system initialized")
else:
    print("❌ Supabase face recognition NOT initialized - check Supabase configuration")

# ===== REDIRECTION HANDLING =====

def handle_redirection_based_on_role(face_data):
    """Handle redirection based on user role with proper URL mapping and status validation"""
    try:
        name = face_data.get('name', 'Unknown')
        role = face_data.get('role', 'unknown')
        confidence = face_data.get('confidence', 0)
        user_id = face_data.get('user_id', 'unknown')
        user_status = face_data.get('status', 'approved')
        
        print(f"🎯 Redirection check: {name} (Role: {role}, Status: {user_status}, Confidence: {confidence}%)")
        
        # Only redirect for confident matches and approved users
        if confidence < 60:
            print("❌ Confidence too low for redirection")
            return None
            
        if user_status != 'approved':
            print(f"❌ User not approved for access. Status: {user_status}")
            return None
        
        # Store user info in session for frontend use
        session['recognized_user'] = {
            'name': name,
            'user_id': user_id,
            'role': role,
            'confidence': confidence,
            'status': user_status,
            'timestamp': datetime.now().isoformat()
        }
        
        # Determine redirection based on role from your database structure
        if role in ['captain', 'official']:  # From barangay_captain and barangay_official
            print(f"🔄 Redirecting {name} to home.html (Role: {role})")
            return '/home'
        elif role == 'response_team':
            print(f"🔄 Redirecting {name} to responsehome.html (Role: {role})")
            return '/responsehome'
        elif role == 'admin':
            print(f"🔄 Redirecting {name} to home.html (Role: {role})")
            return '/home'
        else:
            print(f"ℹ️ No redirection for unknown role: {role}")
            return None
            
    except Exception as e:
        print(f"❌ Error in redirection handling: {e}")
        return None

# ===== SUPABASE UPLOAD FUNCTIONS =====

def upload_to_supabase(image_blob, filename, folder="training-images"):
    """Upload image to Supabase storage with enhanced error handling and URL generation"""
    if not supabase:
        print("❌ Supabase not configured - cannot upload")
        return None, None
    
    try:
        timestamp = int(time.time())
        safe_name = filename.replace(' ', '_').replace('/', '_')
        unique_filename = f"{folder}/{timestamp}_{safe_name}"
        
        print(f"📤 Uploading to Supabase: {unique_filename}")
        
        response = supabase.storage.from_('readytoservethecommunity').upload(
            unique_filename,
            image_blob,
            {"content-type": "image/jpeg", "upsert": "true"}
        )

        if hasattr(response, 'error') and response.error:
            print(f"❌ Upload failed: {response.error}")
            return None, None
        
        public_url = f"https://ptgxqezqawepjaxbqgsx.supabase.co/storage/v1/object/public/readytoservethecommunity/{unique_filename}"
        
        print(f"✅ Upload successful: {public_url}")
        return public_url, unique_filename
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None, None

def store_user_metadata(name, image_urls, image_count, user_data=None):
    """Store user metadata in Firebase under the appropriate role category"""
    try:
        user_id = user_data.get('firebase_user_id') if user_data else str(uuid.uuid4())
        role = user_data.get('role', 'unknown') if user_data else 'unknown'
        timestamp = datetime.now().isoformat()
        
        # Validate and ensure image_urls is properly structured
        validated_image_urls = []
        if image_urls:
            for img_data in image_urls:
                if isinstance(img_data, dict) and 'url' in img_data:
                    validated_image_urls.append({
                        'url': img_data['url'],
                        'storage_path': img_data.get('storage_path', ''),
                        'position': img_data.get('position', 'unknown'),
                        'uploaded_at': timestamp
                    })
                elif isinstance(img_data, str):
                    validated_image_urls.append({
                        'url': img_data,
                        'storage_path': '',
                        'position': 'unknown', 
                        'uploaded_at': timestamp
                    })
        
        print(f"📸 Storing {len(validated_image_urls)} validated image URLs for {name}")
        
        # Session information
        session_info = {
            'session_created': timestamp,
            'last_activity': timestamp,
            'is_active': True,
            'login_count': 1
        }
        
        # Base user metadata for face recognition
        user_metadata = {
            'id': user_id,
            'name': name,
            'image_count': image_count,
            'image_urls': validated_image_urls,
            'created_at': timestamp,
            'updated_at': timestamp,
            'status': 'trained',
            'last_recognition': None,
            'recognition_count': 0,
            'session': session_info
        }
        
        # Add user data if provided
        if user_data:
            user_metadata.update({
                'email': user_data.get('email', ''),
                'firebase_user_id': user_data.get('firebase_user_id', ''),
                'role': user_data.get('role', 'unknown'),
                'trained_at': user_data.get('trained_at', timestamp)
            })
        
        # Store in Firebase under flat 'user' node (not 'users')
        try:
            ref = db.reference(f'/user/{user_id}')
            ref.set(user_metadata)
            print(f"✅ User metadata stored in Firebase user node: {user_id}")
            
            # Enhanced verification
            verify_ref = db.reference(f'/user/{user_id}')
            stored_data = verify_ref.get()
            if stored_data and 'image_urls' in stored_data:
                stored_urls_count = len(stored_data['image_urls'])
                print(f"✅ Verified: {stored_urls_count} URLs stored in Firebase")
                
                if stored_urls_count > 0:
                    sample_url = stored_data['image_urls'][0].get('url', 'No URL')
                    print(f"🔗 Sample image URL: {sample_url}")
                    
                return user_id
            else:
                print("❌ Verification failed: image_urls not found in stored data")
                return user_id
                
        except Exception as e:
            print(f"❌ Firebase storage error: {e}")
            return user_id
        
    except Exception as e:
        print(f"❌ Error storing user metadata: {e}")
        return str(uuid.uuid4())

# ===== API ROUTES =====

@app.route('/api/train', methods=['POST'])
def api_train():
    """Enhanced training endpoint with comprehensive error handling"""
    try:
        print(f"🔧 DEBUG: Supabase available: {supabase is not None}")
        
        # EMERGENCY FALLBACK - If Supabase fails, provide helpful error
        if not supabase:
            error_msg = """
            ❌ Cloud Storage Configuration Error
            
            Supabase is not properly configured. Please check:
            
            1. Your .env file has correct SUPABASE_URL and SUPABASE_KEY
            2. The Supabase project is active and accessible
            3. The 'readytoservethecommunity' bucket exists in storage
            4. Internet connection is working
            
            Current Status:
            - SUPABASE_URL: {url_set}
            - SUPABASE_KEY: {key_set}
            - Supabase Client: {client_initialized}
            
            Contact administrator with this information.
            """.format(
                url_set=bool(os.getenv('SUPABASE_URL')),
                key_set=bool(os.getenv('SUPABASE_KEY')),
                client_initialized=supabase is not None
            )
            
            return jsonify({
                'success': False, 
                'error': 'Cloud storage configuration error. Please contact administrator.',
                'debug_info': {
                    'supabase_url_set': bool(os.getenv('SUPABASE_URL')),
                    'supabase_key_set': bool(os.getenv('SUPABASE_KEY')),
                    'supabase_client_initialized': supabase is not None,
                    'timestamp': datetime.now().isoformat()
                }
            })
        
        name = request.form.get('name')
        files = request.files.getlist('images')
        email = request.form.get('email')
        user_id = request.form.get('userId')
        role = request.form.get('role', 'unknown')

        print(f"🎯 TRAINING START: Name='{name}', UserId='{user_id}', Role='{role}', Files={len(files)}")

        if not name or not files:
            return jsonify({'success': False, 'error': 'Name and images are required'})

        uploaded_urls = []
        saved_count = 0

        # Upload to Supabase
        print("🔄 Uploading images to Supabase...")
        for i, file in enumerate(files):
            if file and file.filename:
                file_data = file.read()
                filename = f"{name.replace(' ', '_')}_{i+1}.jpg"
                
                print(f"📤 Uploading {i+1}/{len(files)}: {filename}")
                public_url, storage_path = upload_to_supabase(file_data, filename)
                
                if public_url and storage_path:
                    uploaded_urls.append({
                        'url': public_url,
                        'storage_path': storage_path,
                        'position': f"position_{i+1}"
                    })
                    saved_count += 1
                    print(f"✅ Upload successful: {public_url}")
                else:
                    print(f"❌ Upload failed: {filename}")

        print(f"📊 Upload complete: {saved_count}/{len(files)} successful")

        if saved_count == 0:
            return jsonify({'success': False, 'error': 'No images processed successfully'})

        # Store metadata in Firebase with proper URL structure
        user_data = {
            'email': email if email else '',
            'firebase_user_id': user_id if user_id else str(uuid.uuid4()),
            'role': role,
            'trained_at': datetime.now().isoformat()
        }
        
        stored_user_id = store_user_metadata(name, uploaded_urls, saved_count, user_data)
        
        # Refresh face recognition cache
        if supabase_face_recognition:
            supabase_face_recognition.refresh_face_cache()

        return jsonify({
            'success': True,
            'message': f'Successfully trained with {saved_count} images for {name}',
            'user_id': stored_user_id,
            'firebase_user_id': user_id,
            'image_urls': uploaded_urls,
            'images_stored_in': 'Supabase Storage'
        })

    except Exception as e:
        print(f"❌ Training error: {e}")
        return jsonify({
            'success': False, 
            'error': f'Training failed: {str(e)}',
            'debug_info': {
                'supabase_status': 'failed',
                'exception_type': type(e).__name__
            }
        })

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    """Single image recognition endpoint using DeepFace Supabase-stored faces with redirection"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'})

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'})

        # Read and process image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'success': False, 'error': 'Invalid image data'})

        # Use DeepFace Supabase-based recognition
        if supabase_face_recognition:
            # Ensure cache is fresh
            supabase_face_recognition.refresh_face_cache()
            results = supabase_face_recognition.recognize_face(image)
        else:
            return jsonify({'success': False, 'error': 'Supabase face recognition not available'})
        
        # Check for redirection based on recognized faces
        redirect_url = None
        recognized_user = None
        
        for result in results:
            if result.get('confidence', 0) > 60 and result.get('status') == 'approved':
                redirect_url = handle_redirection_based_on_role(result)
                if redirect_url:
                    recognized_user = {
                        'name': result.get('name'),
                        'user_id': result.get('user_id'),
                        'role': result.get('role'),
                        'confidence': result.get('confidence'),
                        'status': result.get('status')
                    }
                    break

        response_data = {
            'success': True, 
            'results': results,
            'recognition_type': 'supabase_deepface',
            'timestamp': datetime.now().isoformat()
        }

        # Add redirection info to response
        if redirect_url and recognized_user:
            response_data.update({
                'redirect': True,
                'redirect_url': redirect_url,
                'user': recognized_user
            })
            print(f"🔄 API Response includes redirection to: {redirect_url}")

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Recognition error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/continuous_recognition/start', methods=['POST'])
def start_continuous_recognition():
    """Start continuous recognition with cache refresh"""
    global continuous_recognition_active, frame_count
    
    if not supabase_face_recognition:
        return jsonify({'success': False, 'error': 'Supabase face recognition not available'})
    
    continuous_recognition_active = True
    frame_count = 0
    
    # Force refresh face cache when starting continuous recognition
    supabase_face_recognition.refresh_face_cache()
    
    return jsonify({'success': True, 'message': 'Continuous recognition started with fresh cache'})

@app.route('/api/continuous_recognition/stop', methods=['POST'])
def stop_continuous_recognition():
    """Stop continuous recognition"""
    global continuous_recognition_active
    continuous_recognition_active = False
    return jsonify({'success': True, 'message': 'Continuous recognition stopped'})

@app.route('/api/continuous_recognition/frame', methods=['POST'])
def continuous_recognition_frame():
    """Process a single frame for continuous recognition using DeepFace Supabase recognition"""
    global frame_count, continuous_recognition_active
    
    if not continuous_recognition_active:
        return jsonify({'success': False, 'error': 'Continuous recognition not active'})
    
    if not supabase_face_recognition:
        return jsonify({'success': False, 'error': 'Supabase face recognition not available'})
    
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'})

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'})

        # Read and process image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'success': False, 'error': 'Invalid image data'})

        # Use DeepFace Supabase-based recognition
        results = supabase_face_recognition.recognize_face(image)
        
        # Check for redirection based on recognized faces
        redirect_url = None
        recognized_user = None
        
        for result in results:
            if result.get('confidence', 0) > 60 and result.get('status') == 'approved':
                redirect_url = handle_redirection_based_on_role(result)
                if redirect_url:
                    recognized_user = {
                        'name': result.get('name'),
                        'user_id': result.get('user_id'),
                        'role': result.get('role'),
                        'confidence': result.get('confidence'),
                        'status': result.get('status')
                    }
                    break
        
        frame_count += 1

        response_data = {
            'success': True, 
            'results': results,
            'frame_count': frame_count,
            'recognition_type': 'supabase_deepface',
            'timestamp': datetime.now().isoformat()
        }

        # Add redirection info to response
        if redirect_url and recognized_user:
            response_data.update({
                'redirect': True,
                'redirect_url': redirect_url,
                'user': recognized_user
            })
            print(f"🔄 Continuous recognition redirection to: {redirect_url}")

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Continuous recognition error: {e}")
        return jsonify({'success': False, 'error': str(e)})

# ===== FRONTEND ROUTES =====

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/facial')
def serve_facial():
    return send_from_directory(app.static_folder, 'facial.html')

@app.route('/login')
def serve_login():
    return send_from_directory(app.static_folder, 'login.html')

@app.route('/register')
def serve_register():
    return send_from_directory(app.static_folder, 'register.html')

@app.route('/home')
def serve_home():
    return send_from_directory(app.static_folder, 'home.html')

@app.route('/responsehome')
def serve_responsehome():
    return send_from_directory(app.static_folder, 'responsehome.html')

@app.route('/profile')
def serve_profile():
    return send_from_directory(app.static_folder, 'profile.html')

@app.route('/train')
def serve_train():
    return send_from_directory(app.static_folder, 'train.html')

@app.route('/recognize')
def serve_recognize():
    return send_from_directory(app.static_folder, 'recognize.html')

# ===== HEALTH CHECK =====

@app.route('/api/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint"""
    return jsonify({
        'status': 'OK',
        'message': 'Server is running with DeepFace Supabase face recognition (No local storage)',
        'timestamp': time.time(),
        'continuous_recognition_active': continuous_recognition_active,
        'frame_count': frame_count,
        'supabase_configured': supabase is not None,
        'supabase_face_recognition_configured': supabase_face_recognition is not None,
        'firebase_configured': firebase_admin._apps != [],
        'supabase_faces_loaded': len(supabase_face_recognition.known_faces_cache) if supabase_face_recognition else 0,
        'supabase_embeddings_loaded': sum(len(embeddings) for embeddings in supabase_face_recognition.known_face_embeddings_cache.values()) if supabase_face_recognition else 0,
        'recognition_engine': 'supabase_deepface',
        'model_trained': True,
        'redirection_enabled': True,
        'firebase_structure': {
            'users_node': '/users',
            'recognition_events_node': 'disabled'
        }
    })

# ===== SESSION MANAGEMENT =====

@app.route('/api/session/current', methods=['GET'])
def get_current_session():
    """Get current recognized user from session"""
    try:
        recognized_user = session.get('recognized_user')
        if recognized_user:
            return jsonify({
                'success': True,
                'user': recognized_user
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No user recognized in current session'
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/session/clear', methods=['POST'])
def clear_session():
    """Clear the current session"""
    try:
        session.pop('recognized_user', None)
        return jsonify({
            'success': True,
            'message': 'Session cleared'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("🚀 Starting DeepFace Face Recognition Server (Supabase Only)...")
    print(f"📁 Static folder: {app.static_folder}")
    
    # Debug environment
    print(f"🔧 Environment check:")
    print(f"   FLASK_SECRET_KEY: {'✅' if os.getenv('FLASK_SECRET_KEY') else '❌'}")
    print(f"   SUPABASE_URL: {'✅' if os.getenv('SUPABASE_URL') else '❌'}")
    print(f"   SUPABASE_KEY: {'✅' if os.getenv('SUPABASE_KEY') else '❌'}")
    print(f"   Supabase Client: {'✅' if supabase else '❌'}")
    print(f"   Supabase Face Recognition: {'✅' if supabase_face_recognition else '❌'}")
    
    # Initialize Supabase face recognition cache
    if supabase_face_recognition:
        print("🔄 Initializing Supabase face recognition cache with DeepFace embeddings...")
        supabase_face_recognition.refresh_face_cache()
    else:
        print("❌ Supabase face recognition not available - check your Supabase credentials")
    
    print("✅ Server ready! Access at: http://127.0.0.1:2000")
    print("📱 Facial Recognition Page: http://127.0.0.1:2000/facial")
    print("🎓 Train Faces: http://127.0.0.1:2000/train")
    print("🔍 API Health check: http://127.0.0.1:2000/api/health")
    print("🚫 Local Storage: DISABLED")
    print("🔄 Redirection: Enabled for captain, official → /home")
    print("🔄 Redirection: Enabled for response_team → /responsehome")
    print("🔄 Redirection: Enabled for admin → /home")
    print("🎯 Users will be matched by fullname from Firebase database")

    app.run(debug=False, host='0.0.0.0', port=2000)