# Web Search Integration Guide

## Overview

The RAG system now includes an intelligent orchestrator that decides when to use local documents vs web search, ensuring privacy and security while providing comprehensive answers.

## Features

### 🧠 Intelligent Query Orchestration
- **Privacy Protection**: Automatically detects sensitive information and prevents it from being sent to external APIs
- **Smart Routing**: Decides whether to use local documents, web search, or both based on query analysis
- **User Consent**: Always asks for permission before performing web searches

### 🔒 Privacy & Security Features
1. **Query Filtering**: Removes sensitive information before sending to external APIs
2. **Privacy Levels**: 
   - **Safe**: Can be sent to external APIs
   - **Sensitive**: Requires filtering
   - **Confidential**: Local search only
3. **User Control**: Users can decline web search and use local-only mode

### 🔍 Query Types
- **Local Document**: Questions about uploaded documents
- **Current Events**: Recent news, updates, current information
- **Technical Updates**: Latest versions, releases, documentation
- **General Knowledge**: Public information queries
- **Mixed**: Requires both local and web sources

## Configuration

### API Keys Setup
Add your API keys to `.streamlit/secrets.toml`:

```toml
# Required: Groq API for LLM
groq_api_key = "your_groq_api_key_here"

# Optional: SERP API for web search
serp_api_key = "your_serp_api_key_here"
```

### Get SERP API Key
1. Visit [SerpAPI](https://serpapi.com/)
2. Sign up for a free account (100 searches/month)
3. Get your API key from the dashboard
4. Add it to your secrets.toml file

## Usage Examples

### Example 1: Local Document Query
**User Query**: "What are the key findings in this document?"
- **Decision**: Local search only
- **Reasoning**: Query specifically refers to uploaded document
- **Privacy Level**: Safe (no external search needed)

### Example 2: Current Events Query  
**User Query**: "What are the latest developments in AI regulation?"
- **Decision**: Web search required
- **Reasoning**: Asking for current/latest information
- **Privacy Level**: Safe
- **User Consent**: Required ✅

### Example 3: Mixed Query
**User Query**: "How does this document's approach compare to current industry standards?"
- **Decision**: Hybrid (local + web)
- **Reasoning**: Needs both document content and current industry info
- **Privacy Level**: Sensitive (filtered for web search)
- **User Consent**: Required ✅

### Example 4: Confidential Query
**User Query**: "Based on our internal document, what are the confidential project details?"
- **Decision**: Local search only
- **Reasoning**: Contains confidential information
- **Privacy Level**: Confidential
- **User Consent**: Not required (no external search)

## Privacy Examples

### Original Query vs Filtered Query
- **Original**: "According to our confidential company document, what are the latest TensorFlow features?"
- **Filtered**: "latest TensorFlow features"
- **Removed**: "According to our confidential company document"

## Interface Features

### Search Strategy Toggle
- **Enabled**: Uses intelligent orchestration
- **Disabled**: Traditional local-only search

### Consent Interface
When web search is needed:
1. Shows query analysis and reasoning
2. Displays privacy level and filtered query
3. Offers "Allow Web Search" or "Local Search Only" options
4. Remembers choice for current session

### Enhanced Response Display
- **Query Analysis**: Shows decision reasoning
- **Source Types**: Distinguishes between local documents (📄) and web results (🌐)
- **Strategy Used**: Shows whether local, web, or hybrid approach was used
- **Results Count**: Shows number of local vs web results found

## Security Best Practices

### What Gets Filtered
- Company/organization references ("our company", "internal")
- Confidential keywords ("secret", "private", "confidential")
- Document-specific references ("according to the document")
- System references ("our database", "our system")

### What Stays Local
- Confidential queries (never sent externally)
- Document-specific content analysis
- Sensitive business information
- Personal or proprietary data

### What Can Go External
- General knowledge questions
- Current events queries
- Public technical information
- News and updates (after filtering)

## Benefits

1. **Privacy First**: Your sensitive data never leaves your system
2. **Current Information**: Access to real-time web information when needed
3. **User Control**: Always in control of what gets searched externally
4. **Intelligent Routing**: System automatically chooses the best approach
5. **Transparent Process**: Clear explanation of decisions and reasoning

## Troubleshooting

### Web Search Not Available
- Check if SERP API key is configured in secrets.toml
- Verify API key is valid and has remaining quota
- System will gracefully fall back to local-only search

### Consent Not Appearing
- Ensure "Enable Intelligent Search Strategy" is checked
- Try rephrasing query to trigger web search need
- Check if query is classified as confidential (won't trigger web search)

### Unexpected Local-Only Results
- Query may be classified as confidential
- Check query for sensitive keywords
- Try rephrasing to be more general/public-focused
