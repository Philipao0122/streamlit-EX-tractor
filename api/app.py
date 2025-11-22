from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

# cargar .env
load_dotenv()

# crear temp si no existe
os.makedirs("temp", exist_ok=True)

# inicializar Flask
app = Flask(__name__)
CORS(app)

from analyze_texts.controller import MultiAgentController
from analyze_texts.agent_response import ResponseAgent

# Modo A: reset automático cada vez que se indexan nuevos archivos
controller = MultiAgentController()  # ❌ quitar auto_reset si da error

@app.route("/index-texts", methods=["POST"])
def index_texts():
    try:
        files = request.files.getlist("files")
        if not files:
            return jsonify({"status": "error", "message": "No se enviaron archivos"}), 400

        # 🔥 Reiniciar FAISS antes de procesar nuevos archivos
        print("🧹 Reiniciando índice FAISS...")
        controller.emb_manager.reset_index()
        controller.response_agent = ResponseAgent(faiss_index_path="faiss.index")

        # limpiar carpeta temp
        for f in os.listdir("temp"):
            fp = os.path.join("temp", f)
            if os.path.isfile(fp):
                os.remove(fp)

        file_paths = []
        for f in files:
            dst = os.path.join("temp", f.filename)
            f.save(dst)
            file_paths.append(dst)

        result = controller.process_files(file_paths)
        # result es un dict con analysis y metadata
        return jsonify(result)
    except Exception as e:
        print("❌ Error en /index-texts:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.get_json()
        question = data.get("question", "")
        if not question:
            return jsonify({"status": "error", "message": "No se proporcionó pregunta"}), 400

        answer = controller.answer_question(question)
        return jsonify({"status": "ok", "answer": answer})
    except Exception as e:
        print("❌ Error en /query:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    meta = {
        "status": "ok",
        "total_vectors": controller.emb_manager.index.ntotal if controller.emb_manager and controller.emb_manager.index else 0,
        "total_chunks": len(controller.emb_manager.metadata) if controller.emb_manager and hasattr(controller.emb_manager, "metadata") else 0,
        "groq_configured": bool(os.environ.get("GROQ_API_KEY"))
    }
    return jsonify(meta)

if __name__ == "__main__":
    print("🚀 Servidor iniciado en http://127.0.0.1:5000")
    app.run(debug=True)
