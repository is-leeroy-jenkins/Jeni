# User Guide

Jeni is operated through a Streamlit interface. Launch it from the repository root with:

```powershell
streamlit run app.py
```

Use the sidebar to select the active mode, provider model, runtime parameters, optional system instructions, and mode-specific controls.

## Text Mode

Text mode uses the `Chat` wrapper for Gemini text generation.

Common workflow:

1. Select `Text` mode.
2. Choose a Gemini text model.
3. Enter or load system instructions when needed.
4. Configure inference settings such as temperature, Top-P, Top-K, maximum tokens, stop sequences, and response format.
5. Enable Google Search grounding or URL context only when the response should be grounded in external references.
6. Submit the prompt.
7. Review generated text, sources when available, and token usage.

## Images Mode

Images mode supports generation, analysis, and editing workflows.

| Tab | Purpose |
| --- | --- |
| Generate | Create images from text prompts. |
| Analyze | Upload an image and ask Gemini to analyze it. |
| Edit | Upload an image and provide editing instructions. |

Use model-specific controls for aspect ratio, MIME type, image size, response modality, search grounding, and image-search support where available.

## Audio Mode

Audio mode supports:

| Workflow | Purpose |
| --- | --- |
| Transcribe | Convert uploaded or recorded audio into text. |
| Translate | Translate uploaded or recorded audio into the selected language. |
| Text-to-Speech | Generate audio from text. |

Configure language, voice, sample rate, output format, start time, end time, looping, autoplay, and generation parameters as required by the selected workflow.

## Embedding Mode

Embedding mode converts text into vectors for semantic workflows.

Common workflow:

1. Select an embedding model.
2. Enter or paste source text.
3. Configure chunk size, chunk overlap, dimensions, and encoding format.
4. Normalize and chunk the text.
5. Generate embeddings.
6. Review vector output and text metrics.

## Document Q&A Mode

Document Q&A is a retrieval-augmented workflow.

Supported local document types include:

- PDF
- TXT
- Markdown
- DOCX, where application dependencies support extraction

Workflow:

1. Upload one or more documents.
2. Preview extracted text when available.
3. Build or rebuild the retrieval index.
4. Ask a document-grounded question.
5. Jeni retrieves relevant chunks using `sqlite-vec` when available.
6. If vector-table support is unavailable, Jeni falls back to cosine similarity.
7. Jeni builds a document-grounded prompt and returns an answer.

## Files Mode

Files mode manages Gemini file-oriented workflows, including upload, file ID tracking, metadata review, file purpose/type settings, and file-backed prompt workflows where supported by the selected model.

## File Search Stores Mode

File Search Stores mode supports:

| Workflow | Purpose |
| --- | --- |
| Create | Create a file-search store. |
| Retrieve | Retrieve store metadata. |
| Delete | Delete a selected store. |
| Upload | Upload supported files to the selected store. |

## Google Cloud Buckets Mode

Google Cloud Buckets mode supports cloud bucket creation, retrieval, deletion, and upload workflows. These workflows require valid Google Cloud project and location settings.

## Prompt Engineering Mode

Prompt Engineering mode manages reusable prompts stored in the local SQLite `Prompts` table. Prompt records include `PromptsId`, `Caption`, `Name`, `Text`, `Version`, and `ID`.

Use this mode to search, sort, select, edit, insert, update, delete, and reuse prompt templates in system-instruction areas.

## Data Management Mode

Data Management mode provides local SQLite administration and analysis.

| Tab | Purpose |
| --- | --- |
| Import | Import external data into SQLite. |
| Browse | Browse local SQLite tables. |
| CRUD | Insert, update, and delete rows. |
| Explore | Preview and inspect current data. |
| Filter | Apply advanced filters. |
| Aggregate | Run aggregation operations. |
| Visualize | Build charts from table data. |
| Admin | Drop tables, create tables, create indexes, and alter schema. |
| SQL | Execute guarded read-only SQL queries. |

The SQL validator should be treated as read-oriented. Keep destructive operations out of user-facing query workflows.

## Limitations

- Gemini and Google Cloud features require valid credentials.
- Some controls depend on selected model support.
- PDF extraction depends on PyMuPDF availability.
- Vector indexing uses `sqlite-vec` when available and falls back to cosine similarity when needed.
- `app.py` is not documented with `mkdocstrings` in this drop-in package because it performs Streamlit work during import.
