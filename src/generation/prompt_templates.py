"""
Prompt templates for generation
"""
from typing import List

from src.models import EvidenceSource


class PromptTemplate:
    """Prompt template builder"""
    
    @staticmethod
    def grounded_qa_prompt(
        query: str,
        sources: List[EvidenceSource],
    ) -> str:
        """
        Build a grounded QA prompt
        
        Args:
            query: User query
            sources: Reranked sources
        """
        sources_text = "\n\n".join(
            [f"[Source {source.source_index}]: {source.content}" for source in sources]
        )
        
        prompt = f"""You are a helpful research assistant. Based on the following sources, answer the user's question accurately and cite your sources.

Sources:
{sources_text}

Question: {query}

Use only the supplied evidence. Directly answer the question without copying entire chunks.
Use Markdown and LaTeX only when they improve technical clarity.
Place the exact marker [Source N] immediately after every evidence-supported claim.
Never invent a source number or cite a source not supplied above.
If the evidence is insufficient, say so explicitly instead of guessing.

Answer:"""
        
        return prompt
    
    @staticmethod
    def summarization_prompt(content: str) -> str:
        """Build a summarization prompt"""
        prompt = f"""Please provide a concise summary of the following content in 2-3 sentences:

Content:
{content}

Summary:"""
        return prompt
    
    @staticmethod
    def topic_extraction_prompt(content: str) -> str:
        """Build a topic extraction prompt"""
        prompt = f"""Extract the main topics from the following content. List them as a comma-separated list.

Content:
{content}

Topics:"""
        return prompt
