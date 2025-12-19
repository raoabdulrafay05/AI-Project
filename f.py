import os
import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
from deepface import DeepFace

# --- LIBRARIES FOR VECTOR SEARCH ---
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss

app = Flask(__name__, template_folder="template")

class FaceVectorDB:
    def __init__(self, db_path="face_db_cosine_index"):
        self.db_path = db_path
        self.model_name = "SFace" # Fast model for CPU
        self.embedding_dim = 128  # SFace usually outputs 128 dims
        self.threshold = 0.50     # Cosine Similarity Threshold
        self.vector_store = None
        self._load_or_create_db()

    def _dummy_embed_fn(self, text):
        return np.zeros(self.embedding_dim)

    def _load_or_create_db(self):
        # 1. Try to load existing database
        if os.path.exists(self.db_path):
            print("Loading database...")
            try:
                self.vector_store = FAISS.load_local(
                    self.db_path, 
                    self._dummy_embed_fn,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Error loading DB: {e}. Creating new one.")
                self._create_new_db()
        else:
            self._create_new_db()

    def _create_new_db(self):
        print("Creating new database...")
        index = faiss.IndexFlatIP(self.embedding_dim)
        self.vector_store = FAISS(
            embedding_function=self._dummy_embed_fn,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

    # Normalize vector so Dot Product becomes Cosine Similarity
    def _normalize_vector(self, vector):
        vector = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0: return vector
        return (vector / norm).tolist()

    def generate_embedding(self, image_frame):
        try:
            # DeepFace detects face and returns vector
            results = DeepFace.represent(
                img_path=image_frame, 
                model_name=self.model_name, 
                enforce_detection=True,
                detector_backend="opencv"
            )
            return self._normalize_vector(results[0]["embedding"])
        except:
            return None # No face detected

    # --- SAVE USER ---
    def register_user(self, image_frame, name, doc_id="AUTO"):
        embedding = self.generate_embedding(image_frame)
        if embedding:
            # We only really care about the Name now, but we store ID for technical reasons
            metadata = {"name": name, "docId": doc_id}
            
            self.vector_store.add_embeddings(
                text_embeddings=[(name, embedding)], 
                metadatas=[metadata]
            )
            self.vector_store.save_local(self.db_path)
            return True, f"Registered: {name}"
        else:
            return False, "No face detected in camera."

    # --- IDENTIFY USER ---
    def recognize_user(self, image_frame):
        target_embedding = self.generate_embedding(image_frame)
        if not target_embedding: 
            return {"match": False, "status": "No Face Detected"}

        # Search for closest match
        results = self.vector_store.similarity_search_with_score_by_vector(
            target_embedding, k=1
        )
        
        if not results: 
            return {"match": False, "status": "Unknown"}

        best_match_doc, score = results[0]
        print(f"Match Score: {score}") # Debugging

        if score > self.threshold:
            # Return the name found in database
            found_name = best_match_doc.metadata.get('name', 'Unknown')
            return {"match": True, "name": found_name}
        else:
            return {"match": False, "status": "Unknown Person"}

# Initialize Logic
face_db = FaceVectorDB()

# --- HELPER: CONVERT BROWSER IMAGE TO OPENCV ---
def decode_image(base64_string):
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# --- WEB ROUTES ---

@app.route('/')
def index():
    return render_template('scanner.html')

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    try:
        img = decode_image(data['image'])
        
        success, msg = face_db.register_user(img, data['name'])
        
        return jsonify({'success': success, 'message': msg})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'message': "Server Error"})

@app.route('/api/scan', methods=['POST'])
def scan():
    data = request.json
    try:
        print("Image data received:", data['image'])        

        img = decode_image(data['image'])
        
        # Run recognition
        result = face_db.recognize_user(img)
        
        return jsonify(result)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'match': False, 'status': "Server Error"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)