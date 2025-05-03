"""Implementation of the Model Context Protocol for RAG."""
from typing import List, Dict, Any, Optional, Union, Callable
from pydantic import BaseModel, Field
import json


class ContextItem(BaseModel):
    """A single piece of context that can be provided to an LLM."""
    
    content: str = Field(..., description="The actual content of this context item")
    source_id: Optional[str] = Field(None, description="Unique identifier of the source document")
    source_type: str = Field("document", description="Type of source (document, web, etc.)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about this context")


class ContextSet(BaseModel):
    """A collection of context items with associated metadata."""
    
    items: List[ContextItem] = Field(default_factory=list, description="Collection of context items")
    query: Optional[str] = Field(None, description="The query that led to this context set")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about this context set")
    
    def add_item(self, item: Union[ContextItem, str], **kwargs) -> None:
        """
        Add a context item to the set.
        
        Args:
            item: Either a ContextItem or a string to be converted to one.
            **kwargs: Additional fields if item is a string.
        """
        if isinstance(item, str):
            self.items.append(ContextItem(content=item, **kwargs))
        else:
            self.items.append(item)
    
    def to_formatted_string(self, include_metadata: bool = False) -> str:
        """
        Convert the context set to a formatted string.
        
        Args:
            include_metadata: Whether to include metadata in the output.
            
        Returns:
            Formatted string representation of the context set.
        """
        output = []
        
        if self.query:
            output.append(f"Query: {self.query}\n")
            
        for i, item in enumerate(self.items):
            item_str = f"Context [{i+1}]: {item.content}"
            
            if include_metadata:
                source_info = f" (Source: {item.source_type}"
                if item.source_id:
                    source_info += f", ID: {item.source_id}"
                source_info += ")"
                item_str += source_info
                
                if item.metadata:
                    item_str += f"\nMetadata: {json.dumps(item.metadata, indent=2)}"
                    
            output.append(item_str)
            
        return "\n\n".join(output)
    
    def to_llm_context(self) -> str:
        """
        Format the context set for LLM consumption.
        
        Returns:
            String formatted for LLM context.
        """
        context_parts = []
        
        for i, item in enumerate(self.items):
            context_parts.append(f"[{i+1}] {item.content}")
            
        return "\n\n".join(context_parts)


class ContextProtocolManager:
    """Manager for handling context protocol operations."""
    
    def __init__(self):
        """Initialize the context protocol manager."""
        pass
    
    def from_retrieved_documents(
        self,
        documents: List[Dict[str, Any]],
        query: Optional[str] = None,
        text_key: str = "text",
        metadata_key: str = "metadata",
        id_key: str = "id"
    ) -> ContextSet:
        """
        Create a context set from retrieved documents.
        
        Args:
            documents: List of retrieved documents.
            query: Original query.
            text_key: Key for text in documents.
            metadata_key: Key for metadata in documents.
            id_key: Key for document ID in documents.
            
        Returns:
            ContextSet object.
        """
        context_set = ContextSet(query=query)
        
        for doc in documents:
            context_set.add_item(
                ContextItem(
                    content=doc.get(text_key, ""),
                    source_id=doc.get(id_key),
                    source_type="document",
                    metadata=doc.get(metadata_key, {})
                )
            )
            
        return context_set
    
    def merge_context_sets(self, *context_sets: ContextSet) -> ContextSet:
        """
        Merge multiple context sets into one.
        
        Args:
            *context_sets: Context sets to merge.
            
        Returns:
            Merged context set.
        """
        merged = ContextSet()
        
        for cs in context_sets:
            merged.items.extend(cs.items)
            
            # If there's a query and merged doesn't have one, use it
            if not merged.query and cs.query:
                merged.query = cs.query
                
            # Merge metadata
            for k, v in cs.metadata.items():
                if k not in merged.metadata:
                    merged.metadata[k] = v
        
        return merged
