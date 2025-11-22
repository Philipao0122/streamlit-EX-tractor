# analyze_texts/embeddings.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

class EmbeddingsManager:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", index_path="faiss.index", meta_path="metadata.pkl"):
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.meta_path = meta_path

        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.load_index()
            print(f"✅ Índice cargado: {self.index.ntotal} vectores, {len(self.metadata)} metadatos")
        else:
            # índice plano L2 (funciona bien y simple)
            self.index = faiss.IndexFlatL2(384)
            self.metadata = []
            print("ℹ️ Índice nuevo creado (vacío)")

    def create_embeddings(self, chunks):
        """
        chunks: lista de dicts {'text': '...', 'source': 'file.pdf'}
        """
        if not chunks:
            print("⚠️ No hay chunks para crear embeddings")
            return

        texts = [c["text"] for c in chunks if c.get("text")]
        if not texts:
            print("⚠️ No hay textos válidos en los chunks")
            return

        print(f"📊 Creando embeddings para {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)
        self.metadata.extend(chunks)
        self.save_index()
        print(f"✅ {len(texts)} embeddings creados. Total en índice: {self.index.ntotal}")

    def save_index(self):
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.meta_path, "wb") as f:
                pickle.dump(self.metadata, f)
            print(f"💾 Índice guardado: {self.index.ntotal} vectores")
        except Exception as e:
            print("❌ Error guardando índice:", e)

    def load_index(self):
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)
        except Exception as e:
            print("❌ Error cargando índice:", e)
            self.index = faiss.IndexFlatL2(384)
            self.metadata = []

    def reset_index(self):
        """Borra índice y metadata en memoria y en disco (reset limpio)."""
        self.index = faiss.IndexFlatL2(384)
        self.metadata = []
        try:
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.meta_path):
                os.remove(self.meta_path)
            print("🧹 Índice FAISS reiniciado.")
        except Exception as e:
            print("❌ Error reiniciando archivos de índice:", e)

    def query(self, question, top_k=3):
        if self.index.ntotal == 0:
            print("⚠️ El índice está vacío.")
            return []

        k = min(top_k, self.index.ntotal)
        q_emb = self.model.encode([question])
        D, I = self.index.search(np.array(q_emb).astype("float32"), k)

        results = []
        for idx, dist in zip(I[0], D[0]):
            if 0 <= idx < len(self.metadata):
                results.append(self.metadata[idx])
                print(f"  📄 Resultado distancia={dist:.4f}")
        return results
