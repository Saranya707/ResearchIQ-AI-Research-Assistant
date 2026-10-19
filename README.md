# ResearchIQ — AI Research Assistant Using RAG

A Streamlit application that accepts multiple research PDFs, extracts text with PyMuPDF, creates embeddings with Sentence Transformers, retrieves relevant passages using FAISS, and optionally generates source-grounded answers using an OpenAI model.

## Run locally (Python 3.12 recommended)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed in the terminal (typically http://localhost:8501). Upload text-based PDFs and ask a question.

### AI answer mode

The default **Source passages (free)** mode does not call an LLM. For AI-generated answers, choose the second mode and provide an OpenAI API key in the sidebar, or set `OPENAI_API_KEY` in environment variables or `.streamlit/secrets.toml` (do not commit secrets). API usage may cost money. Relevant document excerpts are sent to the model provider in AI mode.

### Deploy

Push the project files to GitHub, then deploy `app.py` with Streamlit Community Cloud. Configure secrets through the Streamlit deployment settings, not in GitHub. Note that sentence-transformers/FAISS can require significant memory and startup time on small cloud instances.

### Limitations

- Text-based PDFs only; scanned PDFs require OCR.
- Search quality depends on the embedding model, chunking, and document formatting.
- Citations identify source passages and pages but do not guarantee correctness.
- Source-passages mode retrieves evidence; it does not synthesize answers.
- In-memory index is rebuilt when uploads change and is not a persistent multi-user document store.
- For public deployment, add authentication, file-size limits, privacy safeguards, and rate limiting before accepting sensitive files.
