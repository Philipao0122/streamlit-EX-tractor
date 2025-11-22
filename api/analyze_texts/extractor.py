import fitz  # pip install pymupdf
import io
from PIL import Image
import pytesseract

class Extractor:
    def __init__(self):
        # Initialize Tesseract path if needed (uncomment and set your path if necessary)
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pass

    def extract_pdf(self, file_path):
        text = ""
        doc = fitz.open(file_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # First try to extract text directly
            page_text = page.get_text("text").strip()
            
            # If no text or very little text, try OCR on the page image
            if not page_text or len(page_text) < 50:  # Threshold for considering a page as image-based
                try:
                    # Render page to an image
                    pix = page.get_pixmap()
                    img = Image.open(io.BytesIO(pix.tobytes()))
                    # Use Tesseract to do OCR on the image
                    page_text = pytesseract.image_to_string(img, lang='spa+eng')
                except Exception as e:
                    print(f"Error during OCR on page {page_num + 1}: {e}")
            
            if page_text:
                text += page_text + "\n\n"
        
        return text.strip()

    def extract_txt(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encodings if UTF-8 fails
            encodings = ['latin-1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        return f.read()
                except:
                    continue
            raise ValueError(f"No se pudo leer el archivo con ningun encoding: {file_path}")

    def extract_image(self, file_path):
        try:
            img = Image.open(file_path)
            return pytesseract.image_to_string(img, lang='spa+eng')
        except Exception as e:
            print(f"Error al procesar imagen {file_path}: {e}")
            return ""

    def extract(self, file_path):
        try:
            ext = file_path.split('.')[-1].lower()
            if ext == "pdf":
                return self.extract_pdf(file_path)
            elif ext == "txt":
                return self.extract_txt(file_path)
            elif ext in ["png", "jpg", "jpeg", "bmp", "tiff"]:
                return self.extract_image(file_path)
            else:
                raise ValueError(f"Formato no soportado: {ext}")
        except Exception as e:
            print(f"Error en extract() para {file_path}: {e}")
            raise
