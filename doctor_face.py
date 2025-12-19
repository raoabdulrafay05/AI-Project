import cv2
import numpy as np
import os
from deepface import DeepFace

# LangChain & FAISS Imports
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss

class FaceVectorDB:
    def __init__(self, db_path="face_db_cosine_index"):
        self.db_path = db_path
        self.model_name = "SFace"

        self.embedding_dim = self._get_model_dimension()
        
        # --- COSINE THRESHOLD ---
        # 1.00 = Exact Image
        # 0.60 to 0.90 = Same Person
        # Below 0.40 = Different Person
        self.threshold = 0.50 
        
        self.vector_store = None
        self._load_or_create_db()

    def _get_model_dimension(self):
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            embedding = DeepFace.represent(
                img_path=dummy_img,
                model_name=self.model_name,
                enforce_detection=False
            )[0]["embedding"]
            return len(embedding)
        except:
            return 512

    def _load_or_create_db(self):
        print(f"Creating Cosine Database...")
        
        # --- KEY CHANGE: IndexFlatIP ---
        # "IP" stands for Inner Product. 
        # When inputs are normalized, Inner Product == Cosine Similarity.
        index = faiss.IndexFlatIP(self.embedding_dim)
        
        self.vector_store = FAISS(
            embedding_function=self._dummy_embed_fn,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

    def _dummy_embed_fn(self, text):
        return np.zeros(self.embedding_dim)

    def _normalize_vector(self, vector):
        vector = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0: return vector
        return (vector / norm).tolist()

    def generate_embedding(self, image_frame):
        try:
            results = DeepFace.represent(
                img_path=image_frame, 
                model_name=self.model_name, 
                enforce_detection=True,
                detector_backend="opencv"
            )
            return self._normalize_vector(results[0]["embedding"])
        except ValueError:
            return None

    def register_user(self, image_frame, name):
        embedding = self.generate_embedding(image_frame)
        if embedding:
            self.vector_store.add_embeddings(
                text_embeddings=[(name, embedding)], 
                metadatas=[{"name": name}]
            )
            self.vector_store.save_local(self.db_path)
            return True, f"Registered {name}"
        else:
            return False, "No face detected."

    def recognize_user(self, image_frame):
        target_embedding = self.generate_embedding(image_frame)
        if not target_embedding: return "No Face"

        # Search for closest match
        results = self.vector_store.similarity_search_with_score_by_vector(
            target_embedding, k=1
        )
        
        if not results: return "Unknown"

        best_match_doc, score = results[0]
        
        # --- LOGIC FLIPPED ---
        # Higher score is better now.
        print(f"Debug: {best_match_doc.page_content} - Cosine Score: {score:.4f}")

        if score > self.threshold:
            return best_match_doc.page_content
        else:
            return "Unknown"

# --- MAIN APP ---
def run_cpu_optimized_kiosk():
    # Initialize DB (downloads model weights on first run)
    face_db = FaceVectorDB()
    
    cap = cv2.VideoCapture(0)
    # Lower resolution slightly for faster CPU processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("--- CPU Fast Face Recognition ---")
    print("Space: Scan | R: Register | Q: Quit")

    while True:
        ret, frame = cap.read()
        if not ret: break

        cv2.imshow("Hospital Kiosk (CPU)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            name = input("Enter Name: ")
            print(f"Registering {name}...")
            # Capture a few frames to clear buffer
            for _ in range(5): cap.read() 
            ret, clean_frame = cap.read()
            success, msg = face_db.register_user(clean_frame, name)
            print(msg)

        elif key == 32: # Space
            result = face_db.recognize_user(frame)
            print(f"Detected: {result}")
            # Visual Feedback
            color = (0, 255, 0) if result != "Unknown" else (0, 0, 255)
            cv2.putText(frame, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Result", frame)
            cv2.waitKey(1000)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_cpu_optimized_kiosk()