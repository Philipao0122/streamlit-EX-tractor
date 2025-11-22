import os
import re
from groq import Groq
from analyze_texts.embeddings import EmbeddingsManager

def clean_text(text):
    text = re.sub(r'[^\x20-\x7EáéíóúÁÉÍÓÚñÑüÜ.,;:()\-–\[\]{}¿?¡!\\n ]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*\n\s*', '. ', text)
    return text.strip()

class ResponseAgent:
    def __init__(self, faiss_index_path="faiss.index"):
        print("Inicializando ResponseAgent con Groq...")

        # cargar embeddings / metadata
        self.emb_manager = EmbeddingsManager(index_path=faiss_index_path)
        self.index = self.emb_manager.index
        self.metadata = self.emb_manager.metadata

        # groq api key
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "No se encontró GROQ_API_KEY en el entorno."
            )
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.1-8b-instant"

    # ============================================================
    # 🔍 MÉTODO PRINCIPAL PARA RESPONDER PREGUNTAS
    # ============================================================
    def query(self, question, top_k=5):
        results = self.emb_manager.query(question, top_k=top_k)

        if not results:
            return "No encontré información relevante en los documentos cargados."

        context_parts = [clean_text(r["text"]) for r in results]
        context = " ".join(context_parts)[:8000]

        print("\n📝 Contexto recuperado:")
        print(context[:400], "...")

        prompt = f"""
Eres un asistente experto que responde preguntas basándose exclusivamente en el siguiente contexto.

Contexto:
{context}

Pregunta del usuario:
{question}

Da la respuesta más clara posible usando SOLO el contexto.
"""

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_completion_tokens=1200
            )

            return completion.choices[0].message.content

        except Exception as e:
            print("❌ Error en la respuesta:", e)
            return context[:800]

    # ============================================================
    # 🧠 MÉTODO PARA ANALIZAR DOCUMENTOS
    # ============================================================
    def analyze_documents(self, max_chunks=50):
        if self.index.ntotal == 0:
            return "No hay documentos para analizar."

        total = len(self.metadata)
        max_chunks = min(max_chunks, total)

        combined = " ".join([m["text"] for m in self.metadata[:max_chunks]])
        combined = clean_text(combined)

        print(f"🔍 Analizando {max_chunks} chunks...")

        prompt = f"""
Analiza el siguiente contenido extraído de varios documentos.

Contenido:
{combined}

Genera:
1. Resumen general
2. Temas principales
3. Tipo de contenido
4. Puntos relevantes
"""

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_completion_tokens=1500
            )
            return completion.choices[0].message.content

        except Exception as e:
            print(f"❌ Error analizando documentos: {e}")
            return f"Error analizando documentos: {e}\n\nContexto parcial:\n{combined[:800]}"
