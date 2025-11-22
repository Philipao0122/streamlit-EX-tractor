import re
import os
from analyze_texts.extractor import Extractor
from analyze_texts.chunker import Chunker
from analyze_texts.embeddings import EmbeddingsManager
from analyze_texts.agent_response import ResponseAgent


def clean_text(text):
    text = re.sub(r'[^\x20-\x7EáéíóúÁÉÍÓÚñÑüÜ.,;:()\-–\[\]{}¿?¡!\\n ]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*\n\s*', '. ', text)
    return text.strip()


class MultiAgentController:
    def __init__(self, auto_reset=False):
        self.extractor = Extractor()
        self.chunker = Chunker(chunk_size=800, overlap=150)

        # SIEMPRE resetear FAISS cada vez que inicializa el controlador
        self.emb_manager = EmbeddingsManager(index_path="faiss.index")
        self.emb_manager.reset_index()

        self.response_agent = ResponseAgent(faiss_index_path="faiss.index")


    def process_files(self, file_paths):
        print(f"\n{'='*60}")
        print(f"📂 Procesando {len(file_paths)} archivos...")
        print(f"{'='*60}")

        # Reinicia FAISS **cada vez que subes archivos**
        self.emb_manager.reset_index()

        all_chunks = []
        processed_files = []

        for i, fp in enumerate(file_paths, 1):
            try:
                print(f"\n📄 Archivo {i}/{len(file_paths)}: {os.path.basename(fp)}")

                text = self.extractor.extract(fp)
                cleaned_text = clean_text(text)

                if not cleaned_text or len(cleaned_text) < 10:
                    print(f"⚠️ Archivo vacío o muy corto: {fp}")
                    continue

                print(f"  ✅ Texto extraído: {len(cleaned_text)} caracteres")

                raw_chunks = self.chunker.chunk_text(cleaned_text)
                print(f"  ✅ {len(raw_chunks)} chunks creados")

                for chunk in raw_chunks:
                    cleaned_chunk = clean_text(chunk)
                    if cleaned_chunk and len(cleaned_chunk) > 20:
                        all_chunks.append({
                            "text": cleaned_chunk,
                            "source": os.path.basename(fp)
                        })

                processed_files.append(os.path.basename(fp))

            except Exception as e:
                print(f"❌ Error procesando {fp}: {e}")
                continue

        if not all_chunks:
            return {
                "status": "error",
                "message": "❌ No se pudieron extraer chunks válidos",
                "analysis": None
            }

        print(f"\n📊 Total de chunks válidos: {len(all_chunks)}")

        self.emb_manager.create_embeddings(all_chunks)

        # Recargar ResponseAgent con FAISS actualizado
        self.response_agent = ResponseAgent(faiss_index_path="faiss.index")

        # Análisis automático del contenido
        print("\n🤖 Analizando contenido...")
        analysis = self.response_agent.analyze_documents()

        return {
            "status": "success",
            "message": f"✅ {len(all_chunks)} chunks procesados",
            "files_processed": processed_files,
            "total_chunks": len(all_chunks),
            "analysis": analysis
        }


    # -------------- AQUI ESTABA TU ERROR --------------
    def answer_question(self, question):
        """Método usado por /query en app.py"""
        question_cleaned = clean_text(question)
        if not question_cleaned:
            return "❌ La pregunta está vacía."

        return self.response_agent.query(question_cleaned)
