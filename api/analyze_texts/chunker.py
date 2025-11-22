class Chunker:
    def __init__(self, chunk_size=2000, overlap=400):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text):
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.overlap
        return chunks
