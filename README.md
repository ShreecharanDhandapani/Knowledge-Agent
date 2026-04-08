A RAG-powered agent that lets you chat with your entire document library — finding answers and citing sources in seconds.

## Business Problem

Consultants and knowledge workers spend 15–20% of project time searching internal documents, past proposals, and research reports. This agent reduces that to under 2 minutes per query — with source citations so answers are always verifiable.

## Architecture
PDFs → Chunking (1000 chars) → Embeddings (MiniLM-L6) → ChromaDB
↓
User Question → Semantic Retrieval (top-6 chunks) → Llama 3.3 70B → Answer + Sources

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Llama 3.3 70B via Groq API |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | ChromaDB (local persistence) |
| Framework | LangChain |
| UI | Streamlit |

## Key Features

- 📄 Multi-PDF ingestion and indexing
- 🔍 Semantic search across all documents
- 💬 Conversational memory (follow-up questions work)
- 📎 Source citation with file name and page number
- ⚡ ~2 second response time (Groq inference)

## Estimated Impact

| Metric | Before | After |
|---|---|---|
| Document search time | ~20 min | ~2 min |
| Sources checked per query | 2–3 | 50+ |
| Answer traceability | Low | Full citation |

## Project Context

Built as part of an Agentic AI portfolio targeting consulting roles in AI transformation.
