import os
import cv2
import numpy as np
import pickle
import time
import tempfile
from deepface import DeepFace

class FaceUtils:
    def __init__(self):
        print("FaceUtils initialized - Powered by DeepFace")
        self.known_faces = {}
        self.model_file = 'face_data.pkl'
        self.recognition_history = {}
        self.liveness_threshold = 0.6
        
        # DeepFace configuration
        self.model_name = "Facenet"
        self.detector_backend = "opencv"
        self.distance_metric = "cosine"
        self.confidence_threshold = 0.6  # Lower is better match

    def save_training_images(self, name, files):
        """Save training images for a person"""
        print(f"Saving training images for: {name}")
        person_dir = os.path.join('faces', name)
        os.makedirs(person_dir, exist_ok=True)

        saved_count = 0
        for i, file in enumerate(files):
            if file and file.filename:
                filename = f"{name}_{i + 1}.jpg"
                filepath = os.path.join(person_dir, filename)

                # Read image using OpenCV
                image_bytes = file.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is not None:
                    cv2.imwrite(filepath, image)
                    saved_count += 1
                    print(f"✅ Saved: {filename}")
                else:
                    print(f"❌ Failed to save: {filename}")

        print(f"📁 Total images saved for {name}: {saved_count}")
        return saved_count

    def extract_face_embedding(self, image_path):
        """Extract face embedding using DeepFace"""
        try:
            embedding_objs = DeepFace.represent(
                img_path=image_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            
            if embedding_objs and len(embedding_objs) > 0:
                return np.array(embedding_objs[0]['embedding'])
            return None
        except Exception as e:
            print(f"❌ Error extracting embedding: {e}")
            return None

    def train_model(self):
        """Train face recognition model using DeepFace embeddings"""
        print("🚀 Training model with DeepFace...")
        self.known_faces = {}

        if not os.path.exists('faces'):
            print("❌ No faces directory found")
            return {"success": False, "message": "No faces directory"}

        total_embeddings = 0
        total_people = 0
        
        for person_name in os.listdir('faces'):
            person_dir = os.path.join('faces', person_name)
            if os.path.isdir(person_dir):
                embeddings_list = []
                person_images = 0

                for image_file in os.listdir(person_dir):
                    if image_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                        image_path = os.path.join(person_dir, image_file)
                        print(f"🔍 Processing: {image_path}")

                        embedding = self.extract_face_embedding(image_path)
                        if embedding is not None:
                            embeddings_list.append(embedding)
                            total_embeddings += 1
                            person_images += 1
                            print(f"   ✅ Extracted embedding for {person_name} from {image_file}")
                        else:
                            print(f"   ❌ Failed to extract embedding from {image_file}")

                if embeddings_list:
                    self.known_faces[person_name] = embeddings_list
                    total_people += 1
                    print(f"✅ Trained {person_name} with {len(embeddings_list)} face embeddings")

        # Save the model
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.known_faces, f)
            print(f"💾 Model saved to {self.model_file}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return {"success": False, "error": f"Failed to save model: {e}"}

        result = {
            "success": True,
            "message": f"Trained model with {total_people} people and {total_embeddings} embeddings",
            "people_count": total_people,
            "embeddings_count": total_embeddings,
            "model_file": self.model_file
        }
        
        print(f"🎉 Training complete: {result['message']}")
        return result

    def load_model(self):
        """Load trained model with better error handling"""
        try:
            if os.path.exists(self.model_file):
                with open(self.model_file, 'rb') as f:
                    self.known_faces = pickle.load(f)
                
                # Verify the loaded data
                total_embeddings = sum(len(embeddings) for embeddings in self.known_faces.values())
                print(f"✅ Loaded {len(self.known_faces)} known faces with {total_embeddings} DeepFace embeddings")
                return True
            else:
                print("❌ No trained model found - run training first")
                self.known_faces = {}
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.known_faces = {}
            return False

    def compare_faces_deepface(self, embedding1, embedding2):
        """Compare faces using cosine similarity"""
        try:
            if embedding1 is None or embedding2 is None:
                return 1.0  # Maximum distance
            
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 1.0
                
            cosine_similarity = dot_product / (norm1 * norm2)
            distance = 1 - cosine_similarity
            
            return max(0.0, min(1.0, distance))
            
        except Exception as e:
            print(f"❌ Error comparing faces: {e}")
            return 1.0

    def detect_liveness(self, image, face_region):
        """Enhanced liveness detection using image quality metrics"""
        try:
            x, y, w, h = face_region
            face_roi = image[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                return 0.5
            
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            liveness_score = 0.5
            
            # Image quality check (variance)
            variance = np.var(gray_face)
            if variance > 100:  # Good texture
                liveness_score += 0.2
            
            # Face size check
            face_area = w * h
            if 5000 < face_area < 50000:  # Reasonable face size
                liveness_score += 0.2
            
            # Brightness check
            brightness = np.mean(gray_face)
            if 50 < brightness < 200:  # Good lighting
                liveness_score += 0.1
            
            # Edge density (texture)
            edges = cv2.Canny(gray_face, 100, 200)
            edge_density = np.sum(edges > 0) / (w * h)
            if edge_density > 0.05:  # Good edge density
                liveness_score += 0.1
            
            return min(1.0, liveness_score)
            
        except Exception as e:
            print(f"⚠️ Liveness detection error: {e}")
            return 0.5

    def recognize_faces_continuous(self, image, frame_count=0):
        """Continuous face recognition using DeepFace"""
        if not self.load_model():
            print("❌ Cannot recognize faces - model not loaded")
            return []

        current_time = time.time()
        results = []

        try:
            # Save image to temporary file for DeepFace processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                cv2.imwrite(temp_file.name, image)
                temp_file_path = temp_file.name

            try:
                # Detect faces and get embeddings using DeepFace
                embedding_objs = DeepFace.represent(
                    img_path=temp_file_path,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    enforce_detection=False
                )
                
                print(f"🔍 DeepFace found {len(embedding_objs)} faces in frame")

                for embedding_obj in embedding_objs:
                    current_embedding = np.array(embedding_obj['embedding'])
                    facial_area = embedding_obj['facial_area']
                    
                    # Extract face coordinates
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    left, top, right, bottom = x, y, x + w, y + h
                    
                    # Liveness detection
                    liveness_score = self.detect_liveness(image, (x, y, w, h))
                    is_live = liveness_score > self.liveness_threshold
                    
                    # Face recognition
                    best_match_name = "Unknown"
                    best_match_confidence = 0
                    best_distance = 1.0
                    
                    if is_live:  # Only recognize if face is live
                        for person_name, person_embeddings in self.known_faces.items():
                            for known_embedding in person_embeddings:
                                distance = self.compare_faces_deepface(current_embedding, known_embedding)
                                
                                if distance < best_distance:
                                    best_distance = distance
                                    best_match_name = person_name
                        
                        # Convert distance to confidence (0-100%)
                        if best_distance < self.confidence_threshold:
                            confidence = max(0, 100 - (best_distance * 100))
                        else:
                            confidence = 0
                            best_match_name = "Unknown"
                    else:
                        confidence = 0
                        best_match_name = "Unknown"

                    face_id = f"{x}_{y}_{w}_{h}"
                    
                    # Recognition history tracking
                    if face_id not in self.recognition_history:
                        self.recognition_history[face_id] = {
                            'first_seen': current_time,
                            'last_seen': current_time,
                            'recognitions': [],
                            'stable_name': None
                        }

                    self.recognition_history[face_id]['last_seen'] = current_time
                    self.recognition_history[face_id]['recognitions'].append({
                        'name': best_match_name,
                        'confidence': confidence,
                        'timestamp': current_time
                    })

                    # Clean old recognition history
                    for fid in list(self.recognition_history.keys()):
                        if current_time - self.recognition_history[fid]['last_seen'] > 5.0:
                            del self.recognition_history[fid]

                    print(f"📊 Frame {frame_count}: {best_match_name} - Conf: {confidence:.1f}% - Live: {liveness_score:.2f} - Dist: {best_distance:.3f}")

                    results.append({
                        'name': best_match_name,
                        'confidence': float(round(confidence, 1)),
                        'liveness_score': float(round(liveness_score, 2)),
                        'is_live': bool(is_live),
                        'face_id': face_id,
                        'location': {
                            'top': int(top),
                            'right': int(right),
                            'bottom': int(bottom),
                            'left': int(left)
                        },
                        'recognition_type': 'deepface_local'
                    })

            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except Exception as e:
            print(f"❌ DeepFace recognition error: {e}")

        return results

    def recognize_faces(self, image):
        """Single frame recognition with fallback"""
        try:
            return self.recognize_faces_continuous(image)
        except Exception as e:
            print(f"❌ DeepFace recognition failed: {e}")
            # Return empty results instead of crashing
            return []

    def verify_faces(self, image1, image2):
        """Verify if two images contain the same person using DeepFace"""
        try:
            # Save images to temporary files
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp1:
                cv2.imwrite(temp1.name, image1)
                temp1_path = temp1.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp2:
                cv2.imwrite(temp2.name, image2)
                temp2_path = temp2.name

            try:
                # Use DeepFace verify
                result = DeepFace.verify(
                    img1_path=temp1_path,
                    img2_path=temp2_path,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    distance_metric=self.distance_metric,
                    enforce_detection=False
                )
                
                return {
                    'verified': result['verified'],
                    'distance': float(result['distance']),
                    'threshold': float(result['threshold']),
                    'similarity': float(1 - result['distance'])
                }
                
            finally:
                # Clean up
                for path in [temp1_path, temp2_path]:
                    if os.path.exists(path):
                        os.unlink(path)
                        
        except Exception as e:
            print(f"❌ Error in face verification: {e}")
            return {'verified': False, 'error': str(e)}

    def analyze_face(self, image):
        """Analyze face attributes using DeepFace"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                cv2.imwrite(temp_file.name, image)
                temp_file_path = temp_file.name

            try:
                analysis = DeepFace.analyze(
                    img_path=temp_file_path,
                    actions=['age', 'gender', 'emotion', 'race'],
                    detector_backend=self.detector_backend,
                    enforce_detection=False
                )
                
                if analysis and len(analysis) > 0:
                    return analysis[0]
                return None
                
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            print(f"❌ Error in face analysis: {e}")
            return None

    def get_model_info(self):
        """Get information about the current DeepFace model"""
        total_embeddings = sum(len(embeddings) for embeddings in self.known_faces.values())
        return {
            'model_name': self.model_name,
            'detector_backend': self.detector_backend,
            'distance_metric': self.distance_metric,
            'confidence_threshold': self.confidence_threshold,
            'liveness_threshold': self.liveness_threshold,
            'known_people_count': len(self.known_faces),
            'total_embeddings': total_embeddings,
            'model_file': self.model_file,
            'model_loaded': len(self.known_faces) > 0
        }

    def get_known_people(self):
        """Get list of known people"""
        return list(self.known_faces.keys())