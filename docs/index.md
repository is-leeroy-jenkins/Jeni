![](images/jeni_project.png)

#### Jeni Documentation

Jeni is a Python and Streamlit application for building, running, and managing Gemini-powered analytical assistants. It supports text generation, image generation and analysis, image editing, audio transcription, audio translation, text-to-speech, embeddings, document question answering, Gemini file operations, file-search stores, Google Cloud bucket management, prompt engineering, SQLite-backed data management, and export workflows.

The application is designed for federal data analysis, budget execution support, document review, knowledge retrieval, prompt management, and multimodal artificial intelligence experimentation.

## Documentation Map

| Page                                  | Purpose                                                                                                                |
|---------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| [Getting Started](getting-started.md) | Install Jeni, configure the virtual environment, run Streamlit, and build the documentation.                           |
| [Configuration](configuration.md)     | Configure API keys, Google Cloud settings, paths, local storage, and documentation publishing values.                  |
| [User Guide](user-guide.md)           | Use Jeni's text, image, audio, embedding, document, files, stores, bucket, prompt, export, and data modes.             |
| [Architecture](architecture.md)       | Understand the Streamlit shell, Gemini wrappers, configuration layer, persistence, retrieval, and documentation model. |
| [API Reference](api/gemini.md)        | Render Google-style docstrings from import-safe Python modules with mkdocstrings.                                      |

## Core Capabilities

| Capability           | Description                                                                                                          |
|----------------------|----------------------------------------------------------------------------------------------------------------------|
| Text Generation      | Gemini-backed chat and prompt response workflows with optional grounding and URL context.                            |
| Images               | Image generation, analysis, editing, aspect controls, MIME controls, and model-specific options.                     |
| Audio                | Transcription, translation, browser recording support, uploaded audio processing, and text-to-speech.                |
| Embeddings           | Text normalization, chunking, token metrics, embedding generation, and vector inspection.                            |
| Document Q&A         | Local retrieval-augmented question answering using extracted document text and vector search.                        |
| Files and Stores     | Gemini file upload, metadata workflows, file-search stores, and store file upload workflows.                         |
| Google Cloud Buckets | Bucket creation, retrieval, deletion, and upload workflows.                                                          |
| Prompt Engineering   | SQLite-backed reusable prompt records and prompt-template workflows.                                                 |
| Data Management      | SQLite import, browse, CRUD, profiling, filtering, aggregation, visualization, administration, and safe SQL queries. |

## Repository Links

- Source repository: <https://github.com/is-leeroy-jenkins/Jeni>
- Published documentation URL: <https://is-leeroy-jenkins.github.io/Jeni/>
