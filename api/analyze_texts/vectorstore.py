import faiss
import numpy as np
import pickle
import os

class VectorStore:
    def __init__(self, dim=384, index_path="temp/faiss_index"):
        """
        :param dim: dimensión de los embeddings
        :param index_path: ruta para guardar el índice FAISS
        """
        self.dim = dim
        self.index_path = index_path
        self.metadata_path = f"{index_path}_meta.pkl"

        # Si existe índice previo, cargar
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(dim)  # índice plano L2
            self.metadata = []

    def add_embeddings(self, embeddings, metadatas):
        """
        Agrega embeddings y sus metadatos (ej. chunk de texto)
        :param embeddings: np.array de shape (n, dim)
        :param metadatas: lista de strings o dicts con info de cada embedding
        """
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)
        self.metadata.extend(metadatas)

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def query(self, query_embedding, top_k=3):
        """
        Devuelve los top-k chunks más similares
        :param query_embedding: vector np.array (1, dim)
        :return: lista de tuples (score, metadata)
        """
        query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                results.append((dist, self.metadata[idx]))
        return results
