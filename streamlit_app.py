import streamlit as st
import os
import sys
import tempfile
from pathlib import Path
import time
# Add this at the top of the file, right after the imports
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Verify GROQ_API_KEY is loaded
if not os.getenv("GROQ_API_KEY"):
    st.error("Error: GROQ_API_KEY not found in environment variables.")
    st.stop()

# Add the api directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
api_dir = os.path.join(current_dir, 'api')

# Add both the current directory and api directory to the path
for path in [current_dir, api_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from analyze_texts.controller import MultiAgentController
    from analyze_texts.agent_response import ResponseAgent
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.error(f"Current sys.path: {sys.path}")
    st.error(f"Current directory: {os.getcwd()}")
    st.error(f"API directory exists: {os.path.exists(api_dir)}")
    if os.path.exists(api_dir):
        st.error(f"Contents of api directory: {os.listdir(api_dir)}")
    raise

# Configuración de la página
st.set_page_config(
    page_title="Ex-Tractor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
        .main {
            max-width: 1200px;
            margin: 0 auto;
        }
        .stApp {
            background-color: #f5f5f5;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: flex-start;
        }
        .chat-message.user {
            background-color: #29014a;
            margin-left: 20%;
            color: #29014a;
        }
        .chat-message.assistant {
            background-color: #f9f9f9;
            margin-right: 20%;
            color: #29014a;
        }
        .message-avatar {
            margin-right: 1rem;
            font-size: 1.5rem;
        }
        .message-content {
            flex: 1;
        }
        .stTextInput > div > div > input {
            border-radius: 1.5rem;
            padding: 0.75rem 1.5rem;
        }
        .stButton > button {
            border-radius: 1.5rem;
            background: linear-gradient(90deg, #6e48aa 0%, #9d50bb 100%);
            color: white;
            border: none;
            padding: 0.5rem 1.5rem;
            font-weight: 500;
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #5d3a9c 0%, #8c40b3 100%);
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #2d3748;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Inicializar el controlador
@st.cache_resource
def get_controller():
    return MultiAgentController()

controller = get_controller()

# Inicializar el estado de la sesión para los mensajes
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.documents_processed = False
    st.session_state.show_upload = True

# Sidebar para subir archivos
with st.sidebar:
    st.markdown("<h1 style='color: white;'>📂 Documentos</h1>", unsafe_allow_html=True)
    
    if st.session_state.show_upload:
        with st.form("upload-form"):
            uploaded_files = st.file_uploader(
                "Arrastra tus archivos aquí",
                type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            
            if st.form_submit_button("Procesar documentos", type="primary"):
                if uploaded_files:
                    with st.spinner("Procesando documentos..."):
                        # Crear directorio temporal si no existe
                        temp_dir = Path("temp")
                        temp_dir.mkdir(exist_ok=True)
                        
                        # Limpiar archivos temporales anteriores
                        for f in temp_dir.glob("*"):
                            try:
                                f.unlink()
                            except Exception as e:
                                st.error(f"Error al limpiar archivos temporales: {e}")
                        
                        # Guardar archivos subidos
                        file_paths = []
                        for uploaded_file in uploaded_files:
                            file_path = temp_dir / uploaded_file.name
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            file_paths.append(str(file_path))
                        
                        # Procesar archivos
                        try:
                            controller.emb_manager.reset_index()
                            controller.response_agent = ResponseAgent(faiss_index_path="faiss.index")
                            result = controller.process_files(file_paths)
                            st.session_state.documents_processed = True
                            st.session_state.show_upload = False
                            st.session_state.messages.append({"role": "assistant", "content": "¡Documentos procesados exitosamente! ¿En qué puedo ayudarte?"})
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error al procesar documentos: {str(e)}")
                else:
                    st.warning("Por favor, sube al menos un archivo para procesar.")

# Contenedor principal de la aplicación
st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1 style='color: #4a4a4a; margin-bottom: 0.5rem;'>Ex-Tractor Moli</h1>
        <p style='color: #29014a; margin-top: 0;'>Asistente de IA para análisis de documentos</p>
    </div>
""", unsafe_allow_html=True)

# Mostrar mensajes del chat
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.markdown(f"""
                <div class='chat-message user'>
                    <div class='message-avatar'>👤</div>
                    <div class='message-content'>{message["content"]}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='chat-message assistant'>
                    <div class='message-avatar'>🤖</div>
                    <div class='message-content'>{message["content"]}</div>
                </div>
            """, unsafe_allow_html=True)

# Formulario para enviar preguntas
with st.form("question-form", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    with col1:
        question = st.text_input(
            "Escribe tu pregunta aquí...",
            label_visibility="collapsed",
            placeholder="Escribe tu pregunta aquí..."
        )
    with col2:
        submit_btn = st.form_submit_button("Enviar", type="primary")

# Procesar la pregunta
if submit_btn and question.strip():
    # Agregar pregunta al historial
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Obtener respuesta
    with st.spinner("Pensando..."):
        try:
            answer = controller.answer_question(question)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()
        except Exception as e:
            st.error(f"Error al obtener respuesta: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": f"Lo siento, ocurrió un error al procesar tu solicitud: {str(e)}"})
            st.rerun()
elif submit_btn and not question.strip():
    st.warning("Por favor, escribe una pregunta.")

# Mostrar mensaje de bienvenida si no hay mensajes
if not st.session_state.messages:
    st.markdown("""
        <div style='text-align: center; margin: 4rem 0; color: #6b7280;'>
            <h3>¡Hola! Soy tu asistente de IA</h3>
            <p>Sube documentos y hazme preguntas sobre su contenido.</p>
        </div>
    """, unsafe_allow_html=True)
