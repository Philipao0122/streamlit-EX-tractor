from setuptools import setup, find_packages

setup(
    name="extractor_moli",
    version="0.1.0",
    packages=find_packages(where="api"),
    package_dir={"": "api"},
    install_requires=[
        "streamlit==1.34.0",
        "python-dotenv==1.2.1",
        "PyMuPDF==1.26.6",
        "python-docx==1.1.2",
        "sentence-transformers==2.7.0",
        "faiss-cpu==1.12.0",
        "numpy==1.26.4",
        "torch==2.9.1",
        "transformers==4.37.4",
        "pytesseract==0.3.13",
        "Pillow==10.1.0",
        "requests==2.32.5",
        "groq==0.34.1"
    ],
    python_requires=">=3.11",
)
