"""
pipeline/rag.py

Retrieval + confidence check + generation, per the README. Design note on
how "generation" is handled here, since it matters for your hardware setup:

RAGPipeline does NOT load the LLM itself. The actual model (Phi-4-mini-
instruct via transformers + BitsAndBytes NF4 4-bit) gets loaded ONCE at
FastAPI startup (api/main.py) — it's expensive (VRAM, load time) and must
not be reloaded per request or per pipeline instantiation. RAGPipeline
takes a `generate_fn: Callable[[str], str]` in its constructor instead —
you hand it a function that already has the loaded model closed over it.

This keeps two things true at once:
  - rag.py stays testable on a machine with no GPU at all (this sandbox
    included) — you can pass in any stand-in generate_fn.
  - The actual inference call is still a single swappable point, same
    role pipeline/llm.py would have played — it's just that here it's a
    function you write once in api/main.py and inject, rather than a
    separate importable file. If you'd rather have that as its own file
    after all, say so and I'll split it out.

Retrieval + confidence fields returned by generate() map directly onto
the `interactions` table columns (retrieval_top1_id, retrieval_top1_score,
retrieval_top3_avg_score, low_confidence_flag) — api/routes/suggest.py
should write these straight into the DB alongside the model's answer.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from pipeline.embed import Embedder, load_index


@dataclass
class RetrievalHit:
    score: float
    metadata: dict


@dataclass
class RAGResult:
    query: str
    suggestion: str
    retrieved: list[RetrievalHit]
    retrieval_top1_id: Optional[str]
    retrieval_top1_score: Optional[float]
    retrieval_top3_avg_score: Optional[float]
    low_confidence_flag: bool
    prompt: str = field(repr=False)  # kept for logging/debugging, excluded from default repr


class RAGPipeline:
    """
    Usage (api/main.py, once at startup):
        embedder = Embedder()
        index, metadata = load_index()

        def generate_fn(prompt: str) -> str:
            # closes over the already-loaded Phi-4-mini model + tokenizer
            return model.generate(...)

        rag = RAGPipeline(embedder, index, metadata, generate_fn)

    Usage (per request, api/routes/suggest.py):
        result = rag.generate(incoming_customer_text)
    """

    def __init__(self, embedder: Embedder, index, metadata: list[dict],
                 generate_fn: Callable[[str], str],
                 top_k: int = 3, confidence_threshold: float = 0.45):
        self.embedder = embedder
        self.index = index
        self.metadata = metadata
        self.generate_fn = generate_fn
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold

    # --- Retrieval ---------------------------------------------------

    def retrieve(self, query_text: str, k: int = None) -> list[RetrievalHit]:
        k = k or self.top_k
        k = min(k, self.index.ntotal)  # guard: don't ask FAISS for more than exists
        if k == 0:
            return []

        query_vec = self.embedder.embed([query_text]).astype("float32")
        scores, indices = self.index.search(query_vec, k)

        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS pads with -1 if fewer than k results exist
            hits.append(RetrievalHit(score=float(score), metadata=self.metadata[idx]))
        return hits

    # --- Confidence scoring --------------------------------------------

    def _confidence_fields(self, hits: list[RetrievalHit]):
        if not hits:
            return None, None, None, True  # no hits at all = definitely low confidence

        top1 = hits[0]
        top3 = hits[:3]
        top3_avg = sum(h.score for h in top3) / len(top3)
        low_confidence = top1.score < self.confidence_threshold

        return top1.metadata.get("id"), top1.score, top3_avg, low_confidence

    # --- Prompt construction ---------------------------------------------

    def build_prompt(self, query_text: str, hits: list[RetrievalHit]) -> str:
        """
        Kept as its own method (not inlined into generate()) so
        prompt_templates can evolve independently and so tests can assert
        on prompt content without invoking the model.
        """
        if not hits:
            context_block = "No relevant past tickets or KB articles were found."
        else:
            context_lines = []
            for hit in hits:
                m = hit.metadata
                if m["type"] == "ticket":
                    context_lines.append(
                        f"- Past ticket (similarity {hit.score:.2f}): "
                        f"Customer said: \"{m['customer_text']}\" | "
                        f"Agent resolved with: \"{m['agent_response']}\""
                    )
                elif m["type"] == "kb_doc":
                    context_lines.append(
                        f"- KB article \"{m['title']}\" (similarity {hit.score:.2f}): {m['content']}"
                    )
            context_block = "\n".join(context_lines)

        return (
            "You are a customer support assistant. Draft a helpful, concise reply "
            "to the customer's message below, using the reference context if relevant. "
            "Do not invent policies or details not supported by the context.\n\n"
            f"Reference context:\n{context_block}\n\n"
            f"Customer message:\n{query_text}\n\n"
            "Draft reply:"
        )

    # --- End-to-end -------------------------------------------------------

    def generate(self, query_text: str) -> RAGResult:
        hits = self.retrieve(query_text)
        top1_id, top1_score, top3_avg, low_confidence = self._confidence_fields(hits)
        prompt = self.build_prompt(query_text, hits)
        suggestion = self.generate_fn(prompt)

        return RAGResult(
            query=query_text,
            suggestion=suggestion,
            retrieved=hits,
            retrieval_top1_id=top1_id,
            retrieval_top1_score=top1_score,
            retrieval_top3_avg_score=top3_avg,
            low_confidence_flag=low_confidence,
            prompt=prompt,
        )