from pipeline import PipelineStep

class TokenChunker(PipelineStep):
    """Chunker that splits on a token basis."""
    def __init__(
        self, chunk_size: int, overlap: int = 0, encoding_name: str = "cl100k_base"
    ):
        try:
            import tiktoken
            self.tiktoken = tiktoken
        except ImportError:
            raise ImportError(
                "The `tiktoken` library must be installed to use `TokenChunker`. "
                "To install it, run the following command: `pip install tiktoken`"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding_name = encoding_name

    def forward(self, data):
        encoding = self.tiktoken.get_encoding(self.encoding_name)
        tokens = encoding.encode(data, disallowed_special=())
        chunks = [
            encoding.decode(tokens[i:i + self.chunk_size])
            for i in range(0, len(tokens), self.chunk_size - self.overlap)
        ]
        return chunks
    
class SentenceChunker(PipelineStep):
    """Chunker that splits on a sentence basis."""
    def __init__(
        self, chunk_size: int, overlap: int = 0, model: str = "punkt_tab"
    ):
        try:
            import nltk
            self.nltk = nltk
            self.nltk.download(model, quiet=True)
        except ImportError:
            raise ImportError(
                "The `nltk` library must be installed to use `SentenceChunker`. "
                "To install it, run the following command: `pip install nltk`"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap

    def forward(self, data):
        sentences = self.nltk.tokenize.sent_tokenize(data)
        chunks = [
            ' '.join(sentences[i:i + self.chunk_size])
            for i in range(0, len(sentences), self.chunk_size - self.overlap)
        ]
        return chunks
    
class SemanticChunker(PipelineStep):
    """Chunker that splits on the semantic similarity of consecutive sentences."""
    def __init__(
        self,
        overlap: int = 0,
        similarity_threshold: float = 0.3,
        model: str = "all-MiniLM-L6-v2",
        nltk_model: str = "punkt_tab"
    ):
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            import nltk

            self.model = SentenceTransformer(model)
            self.np = np
            self.nltk = nltk
            self.nltk.download(nltk_model, quiet=True)
        except:
            raise ImportError(
                "`sentence_transformers`, `numpy`, and `nltk` must be "
                "installed to use `SemanticChunker`. Run the following command to "
                "install them: `pip install sentence_transformers numpy nltk`"
            )
        self.overlap = overlap
        self.similarity_threshold = similarity_threshold
        
    def _create_chunks(self, sentences: list[str], embeddings) -> list[str]:
        chunks = []
        current_chunk = []
        for i in range(len(embeddings) - 1):
            current_chunk.append(sentences[i])
            current_sentence, next_sentence = embeddings[i], embeddings[i + 1]
            similarity = self.model.similarity(current_sentence, next_sentence)
            if similarity < self.similarity_threshold:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        current_chunk.append(sentences[-1])
        chunks.append(" ".join(current_chunk))
        return chunks
        

    def forward(self, data):
        sentences = self.nltk.tokenize.sent_tokenize(data)
        embeddings = self.model.encode(sentences)
        chunks = self._create_chunks(sentences, embeddings)
        return chunks