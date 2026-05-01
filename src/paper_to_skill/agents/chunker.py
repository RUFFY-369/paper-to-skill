import re

class StructuralChunker:
    """
    Stage A.5: Structural Chunking and Routing for high-efficiency and zero-bloat.
    """
    def __init__(self):
        self.relevant_keywords = [
            "tensor", "shape", "dtype", "hardware", "quantize", "quantization",
            "cuda", "sm_", "dimension", "objective", "mean", "outlier", "formula",
            "tiling", "block", "precision", "int8", "int4", "fp8", "exact attention",
            "sage", "input", "output", "gemm"
        ]

    def chunk_and_route(self, text: str) -> str:
        """
        Splits input text by markdown headers and routes/joins only highly relevant chunks.
        """
        if not text:
            return ""

        # Break text by markdown sections using header indicators
        sections = re.split(r"(?m)^(?:#{1,4}\s+.*$)", text)
        headers = re.findall(r"(?m)^(?:#{1,4}\s+.*$)", text)

        chunks = []
        if sections and sections[0].strip():
            chunks.append(sections[0].strip())

        for i, h in enumerate(headers):
            if i + 1 < len(sections):
                chunks.append(h + "\n" + sections[i + 1].strip())
            else:
                chunks.append(h)

        # Prune sections matching References, Related Work, and Acknowledgements
        prune_keywords = ["references", "related work", "acknowledgements", "acknowledgments"]

        # Route only relevant chunks based on keyword heuristics
        filtered_chunks = []
        for c in chunks:
            if any(pkw in c.lower() for pkw in prune_keywords):
                continue
            if any(kw in c.lower() for kw in self.relevant_keywords):
                filtered_chunks.append(c)

        # If everything is filtered out, fallback to original text
        if not filtered_chunks:
            return text

        return "\n\n".join(filtered_chunks)

