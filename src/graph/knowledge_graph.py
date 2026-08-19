"""
Citation and knowledge graph
"""
from typing import Dict, List, Set

import networkx as nx

from src.logging_config import get_logger
from src.models import Citation

logger = get_logger(__name__)


class KnowledgeGraph:
    """Knowledge graph for citations and relationships"""
    
    def __init__(self):
        """Initialize knowledge graph"""
        self.graph = nx.DiGraph()
    
    def add_citation_relation(
        self,
        source_doc_id: str,
        target_doc_id: str,
        citation,
    ) -> None:
        """
        Add a citation relation between documents
        
        Args:
            source_doc_id: Document containing the citation
            target_doc_id: Document being cited
            citation: Citation object or string description
        """
        # Add nodes if not exist
        self.graph.add_node(source_doc_id, type="document")
        self.graph.add_node(target_doc_id, type="document")
        
        # Handle both Citation objects and strings
        if isinstance(citation, str):
            citation_score = 1.0
            content = citation
        else:
            citation_score = citation.relevance_score
            content = citation.content
        
        # Add edge with citation data
        self.graph.add_edge(
            source_doc_id,
            target_doc_id,
            citation_score=citation_score,
            content=content,
        )
        
        logger.debug(f"Added citation: {source_doc_id} -> {target_doc_id}")
    
    def get_cited_documents(self, doc_id: str) -> List[str]:
        """Get all documents cited by a document"""
        return list(self.graph.successors(doc_id))
    
    def get_citing_documents(self, doc_id: str) -> List[str]:
        """Get all documents that cite a document"""
        return list(self.graph.predecessors(doc_id))
    
    def get_citation_score(self, source_doc_id: str, target_doc_id: str) -> float:
        """Get citation score between two documents"""
        if self.graph.has_edge(source_doc_id, target_doc_id):
            return self.graph[source_doc_id][target_doc_id].get("citation_score", 0.0)
        return 0.0
    
    def compute_importance_scores(self) -> Dict[str, float]:
        """
        Compute importance scores for documents using PageRank
        
        Returns:
            Dictionary of document_id to importance score
        """
        if not self.graph:
            return {}
        
        # Use PageRank to compute importance
        scores = nx.pagerank(self.graph, alpha=0.85)
        return scores
    
    def find_citation_paths(
        self,
        source_doc_id: str,
        target_doc_id: str,
    ) -> List[List[str]]:
        """
        Find all citation paths from source to target
        
        Args:
            source_doc_id: Starting document
            target_doc_id: Target document
        """
        try:
            paths = list(nx.all_simple_paths(self.graph, source_doc_id, target_doc_id))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def get_related_documents(
        self,
        doc_id: str,
        depth: int = 2,
    ) -> Set[str]:
        """
        Get documents related within a certain depth
        
        Args:
            doc_id: Starting document
            depth: Maximum depth to search
        """
        related = set()
        visited = {doc_id}
        frontier = {doc_id}
        
        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                # Get neighbors
                neighbors = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
                for neighbor in neighbors:
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        related.add(neighbor)
                        visited.add(neighbor)
            frontier = next_frontier
        
        return related

    def expand_documents(
        self,
        doc_ids: List[str],
        direction: str = "both",
        depth: int = 1,
    ) -> Set[str]:
        """Expand seed documents through upstream, downstream, or both citation edges."""
        if direction not in {"upstream", "downstream", "both"}:
            raise ValueError("direction must be 'upstream', 'downstream', or 'both'")
        if depth < 1:
            raise ValueError("depth must be at least 1")

        visited = set(doc_ids)
        frontier = set(doc_ids)
        expanded = set()
        for _ in range(depth):
            next_frontier = set()
            for doc_id in frontier:
                if doc_id not in self.graph:
                    continue
                neighbors = set()
                if direction in {"downstream", "both"}:
                    neighbors.update(self.graph.successors(doc_id))
                if direction in {"upstream", "both"}:
                    neighbors.update(self.graph.predecessors(doc_id))
                next_frontier.update(neighbors - visited)
            expanded.update(next_frontier)
            visited.update(next_frontier)
            frontier = next_frontier
        return expanded
