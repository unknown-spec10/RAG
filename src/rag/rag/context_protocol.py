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
    """
    Enhanced ContextSet supporting dynamic sizing, prioritization, and condensation.
    """

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
    
    def token_length(self, llm_tokenizer) -> int:
        """
        Estimate token length of the context set using the provided tokenizer.
        Args:
            llm_tokenizer: Callable or tokenizer object with encode method
        Returns:
            Total token count
        """
        text = self.to_llm_context()
        return len(llm_tokenizer.encode(text))

    def prioritize_and_truncate(self, llm_tokenizer, max_tokens: int, query: str = None, expected_response_tokens: int = 512) -> "ContextSet":
        """
        Prioritize and truncate context items to fit within the available token window.
        Args:
            llm_tokenizer: Tokenizer for token counting
            max_tokens: Total LLM context window
            query: The user query (for relevance)
            expected_response_tokens: Reserve this many tokens for the LLM's response
        Returns:
            New ContextSet with prioritized and truncated items
        """
        # Simple prioritization: core facts > background > counter-arguments > other
        priority_map = {"core": 0, "background": 1, "counter": 2, "other": 3}
        def get_priority(item):
            t = item.metadata.get("context_type", "other")
            return priority_map.get(t, 3)
        sorted_items = sorted(self.items, key=get_priority)
        # Iteratively add items until token limit is reached
        new_set = ContextSet(query=self.query, metadata=self.metadata.copy())
        used_tokens = len(llm_tokenizer.encode(query or "")) + expected_response_tokens
        for item in sorted_items:
            candidate_set = ContextSet(items=new_set.items + [item], query=self.query, metadata=self.metadata.copy())
            tokens = candidate_set.token_length(llm_tokenizer)
            if tokens + used_tokens > max_tokens:
                break
            new_set.items.append(item)
        return new_set

    def condense(self, llm, target_tokens: int, llm_tokenizer=None) -> "ContextSet":
        """
        Summarize/condense the least important context to fit within target_tokens.
        Args:
            llm: LLM or summarization model (must have invoke or __call__)
            target_tokens: Target total tokens for context
            llm_tokenizer: Optional tokenizer for token counting
        Returns:
            New ContextSet with condensed items
        """
        if llm_tokenizer is None:
            def count_tokens(text): return len(text.split())
        else:
            def count_tokens(text): return len(llm_tokenizer.encode(text))
        # If already fits, return self
        if self.token_length(llm_tokenizer or (lambda x: x.split())) <= target_tokens:
            return self
        # Condense least important items
        sorted_items = sorted(self.items, key=lambda x: x.metadata.get("context_type", "other"))
        # Summarize the last N items until fits
        items = list(sorted_items)
        while items and ContextSet(items=items, query=self.query, metadata=self.metadata.copy()).token_length(llm_tokenizer or (lambda x: x.split())) > target_tokens:
            # Take the least important item(s)
            to_summarize = items[-1]
            summary = llm.invoke(f"Summarize the following for context compression:\n{to_summarize.content}")
            items[-1] = ContextItem(content=summary if isinstance(summary, str) else getattr(summary, 'content', str(summary)),
                                   source_id=to_summarize.source_id,
                                   source_type=to_summarize.source_type,
                                   metadata=to_summarize.metadata)
            # Optionally merge multiple items, or drop if still too long
            if count_tokens(items[-1].content) < 10:
                items.pop(-1)
        return ContextSet(items=items, query=self.query, metadata=self.metadata.copy())

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
    """
    Enhanced ContextProtocolManager supporting protocol refinement and dynamic context sizing.
    """

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
    
    def refine_protocol(self, context_set: ContextSet, protocol: str = "default") -> ContextSet:
        """
        Refine context protocol by reordering or filtering context items according to protocol.
        Args:
            context_set: The ContextSet to refine
            protocol: Protocol type (e.g., 'default', 'core-first', 'background-minimal')
        Returns:
            Refined ContextSet
        """
        if protocol == "core-first":
            # Move core facts to front
            core = [i for i in context_set.items if i.metadata.get("context_type") == "core"]
            rest = [i for i in context_set.items if i.metadata.get("context_type") != "core"]
            return ContextSet(items=core + rest, query=context_set.query, metadata=context_set.metadata.copy())
        elif protocol == "background-minimal":
            # Drop background info
            filtered = [i for i in context_set.items if i.metadata.get("context_type") != "background"]
            return ContextSet(items=filtered, query=context_set.query, metadata=context_set.metadata.copy())
        # Default: no change
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
