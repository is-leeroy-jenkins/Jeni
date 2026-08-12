###### Jeni

![](https://github.com/is-leeroy-jenkins/Jeni/blob/main/resources/images/jeni_project.png)

<p align="center">
  <a href="#-overview">Overview</a> 
  &bull;
  <a href="#-features">Features</a>
  &bull;
  <a href="#-application-modes">Modes</a>
  &bull;
  <a href="https://github.com/is-leeroy-jenkins/Jeni/blob/main/requirements.txt">Requirements</a> 
  &bull;
  <a href="#-api-key-setup">Setup</a> 
  &bull;
  <a href="#-installation">Installation</a> 
  &bull;
  <a href="#-running-the-streamlit-application">Run</a> 
  &bull;
  <a href="#-configuration">Configuration</a> 
  &bull;
  <a href="#-design-and-architecture">Architecture</a>
  &bull;
  <a href="#-capabilities">Capabilities</a> 
  &bull;
</p>

___


[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-0078FC?style=for-the-badge&logo=githubpages&logoColor=white)](https://is-leeroy-jenkins.github.io/Jeni/)


Jeni is a Python and Streamlit application for building, running, and managing
Gemini-powered analytical assistants. It supports text generation, image generation and
analysis, image editing, audio transcription, audio translation, text-to-speech, embeddings,
document question answering, Gemini file operations, file-search stores, Google Cloud bucket
management, prompt engineering, SQLite-backed data management, and export workflows.

Jeni is designed for federal data analysis, budget execution support, document review,
knowledge retrieval, prompt management, and multimodal artificial intelligence experimentation.

## 🎥 Demo
![](https://github.com/is-leeroy-jenkins/Jeni/blob/main/resources/images/jeni-demo.gif)

___

## 🧊 Azure

[![Containerized](https://img.shields.io/badge/Docker-App-2496ED?logo=docker&logoColor=white)](https://jeni.grayrock-3f318ce3.eastus.azurecontainerapps.io)

- Containerized app

## 🔥 Streamlit

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://jeni-py.streamlit.app/)

- Web app

## 🧱 Databricks

[![Jeni](https://img.shields.io/badge/Databricks-Jeni-FF3621?logo=databricks\&logoColor=white)](https://dbc-a0c21f80-7bb3.cloud.databricks.com/browse/folders/3169291152438615?o=7474645703081351)

* Databricks workspace repository for the Jeni codebase.
* Supports collaborative development, analytics, notebook execution, and application deployment.


## 🧰 Overview

- Core classes imported by the application include:

| Class           | Purpose                                                                    |
| --------------- | -------------------------------------------------------------------------- |
| `Chat`          | Text generation and Google-grounded Gemini chat workflows                  |
| `Images`        | Image generation, image analysis, image editing, and image search support  |
| `Embeddings`    | Text embedding generation and vector preparation                           |
| `Transcription` | Audio-to-text transcription                                                |
| `Translation`   | Audio translation workflows                                                |
| `TTS`           | Text-to-speech generation                                                  |
| `Files`         | Gemini Files API workflows                                                 |
| `FileSearch`    | File-search store creation, retrieval, deletion, and file upload workflows |
| `CloudBuckets`  | Google Cloud bucket creation, retrieval, deletion, and upload workflows    |

## ✨ Features

* **Gemini-first interface** for text, image, audio, document, embedding, file, and cloud workflows.
* **Single Streamlit application** with explicit mode selection in the sidebar.
* **Text generation controls** for model selection, temperature, Top-P, Top-K, frequency penalty,
  presence penalty, output format, response schema, safety profile, stop sequences, stream mode,
  Google Search grounding, and URL context.
* **Image workflows** for generation, analysis, and editing with aspect ratio, output MIME type,
  response modality, Google Search grounding, and image-search options where supported.
* **Audio workflows** for transcription, translation, and text-to-speech using uploaded audio files
  or recorded browser audio.
* **Document Q&A** with local document loading, text extraction, chunking, embeddings, and
  SQLite vector retrieval with fallback cosine similarity.
* **Embedding mode** with text normalization, chunking, token metrics, vector display, and
  configurable embedding dimensions and encoding format.
* **Files API mode** for server-side file upload and file operations.
* **File Search Stores mode** for creating, retrieving, deleting, and uploading files to file-search
  stores.
* **Google Cloud Buckets mode** for creating, retrieving, deleting, and uploading files to cloud
  bucket-backed workflows.
* **Prompt Engineering mode** backed by a local SQLite `Prompts` table.
* **Data Export mode** for exporting prompt and data assets.
* **Data Management mode** for SQLite import, browsing, CRUD operations, profiling, filtering,
  aggregation, visualization, administration, and safe SQL queries.
* **Footer status bar** showing provider, mode, model, and active runtime settings.

## 🧩 Application Modes

| Mode                   | Description                                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------------------------------ |
| `Text`                 | Gemini text generation with system prompts, templates, grounding, URL context, schema output, and streaming. |
| `Images`               | Image generation, image analysis, and image editing using Gemini image-capable models.                       |
| `Audio`                | Audio transcription, audio translation, browser recording, uploaded audio processing, and text-to-speech.    |
| `Embedding`            | Text normalization, chunking, embedding generation, metrics, and vector inspection.                          |
| `Document Q&A`         | Upload and preview documents, build a local retrieval index, and ask document-grounded questions.            |
| `Files`                | Manage Gemini Files API operations and related file metadata.                                                |
| `File Search Stores`   | Create, retrieve, delete, and upload files to file-search stores.                                            |
| `Google Cloud Buckets` | Create, retrieve, delete, and upload files to Google Cloud bucket-backed workflows.                          |
| `Prompt Engineering`   | Manage reusable prompts in the local SQLite `Prompts` table.                                                 |
| `Data Export`          | Export prompts, system instructions, and local data assets.                                                  |
| `Data Management`      | Import, browse, edit, profile, filter, aggregate, visualize, administer, and query SQLite data.              |

## 🛠️ Requirements

| Requirement                            | Purpose                                                                    |
| -------------------------------------- | -------------------------------------------------------------------------- |
| Python 3.10+                           | Runtime environment                                                        |
| Streamlit                              | Web application framework                                                  |
| google-genai / Gemini SDK dependencies | Gemini client access                                                       |
| pandas                                 | DataFrame operations and SQLite table display                              |
| numpy                                  | Vector math and cosine similarity                                          |
| plotly                                 | Interactive visualizations                                                 |
| tiktoken                               | Token counting                                                             |
| sentence-transformers                  | Local document embedding model                                             |
| sqlite-vec                             | SQLite vector table support                                                |
| PyMuPDF / `fitz`                       | PDF text extraction                                                        |
| reportlab                              | PDF export support                                                         |
| boogr                                  | Application error handling                                                 |
| config.py                              | Application constants, model lists, paths, labels, and API defaults        |
| SQLite                                 | Local persistence for prompts, chat history, embeddings, and imported data |

## 🔑 API Key Setup

Jeni reads API and cloud configuration from `config.py`, environment variables, and Streamlit session
state. Sidebar-entered values override configuration defaults for the current session.

| Key / Setting             | Used For                                           |
| ------------------------- | -------------------------------------------------- |
| `GEMINI_API_KEY`          | Gemini API access                                  |
| `GOOGLE_API_KEY`          | Google API access and Gemini API-key mode fallback |
| `GOOGLE_CSE_ID`           | Google Custom Search integration                   |
| `GOOGLEMAPS_API_KEY`      | Google Maps-related workflows                      |
| `GOOGLE_CLOUD_PROJECT_ID` | Google Cloud project routing                       |
| `GOOGLE_CLOUD_LOCATION`   | Google Cloud regional configuration                |

Helpful setup references:

* [Gemini API Key](https://github.com/is-leeroy-jenkins/Buddy/blob/main/resources/setup/gemini.md)

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/is-leeroy-jenkins/Jeni.git
cd Jeni
```

### 2. Create and Activate a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## ⚙️ Configuration

Set the required values in `config.py`, environment variables, or the Streamlit sidebar.

Example environment variables:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export GOOGLE_API_KEY="your-google-api-key"
export GOOGLE_CSE_ID="your-google-custom-search-id"
export GOOGLEMAPS_API_KEY="your-google-maps-api-key"
export GOOGLE_CLOUD_PROJECT_ID="your-google-cloud-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
```

Windows PowerShell:

```powershell
setx GEMINI_API_KEY "your-gemini-api-key"
setx GOOGLE_API_KEY "your-google-api-key"
setx GOOGLE_CSE_ID "your-google-custom-search-id"
setx GOOGLEMAPS_API_KEY "your-google-maps-api-key"
setx GOOGLE_CLOUD_PROJECT_ID "your-google-cloud-project-id"
setx GOOGLE_CLOUD_LOCATION "us-central1"
```

## 🚀 Running the Streamlit Application

From the project root:

```bash
streamlit run app.py
```

Once running, the application is available at:

```text
http://localhost:8501
```

## 🧠 Text Generation

The `Text` mode provides Gemini chat and text generation through the `Chat` wrapper.

Supported controls include:

| Control Group                | Options                                                                                           |
| ---------------------------- | ------------------------------------------------------------------------------------------------- |
| Model Settings               | Model, thinking level, response modalities, media resolution, candidates                          |
| Inference Settings           | Temperature, Top-P, Top-K, frequency penalty, presence penalty                                    |
| Grounding / Context Settings | Google Search grounding, URL list, maximum URLs                                                   |
| Output / Response Settings   | Max tokens, response MIME type, JSON response schema, stop sequences, safety profile, stream mode |
| System Prompt                | Manual system instructions, prompt-template loading, clear button, XML-to-Markdown conversion     |

## 📷 Images

The `Images` mode supports three tabs:

| Tab      | Purpose                                          |
| -------- | ------------------------------------------------ |
| Generate | Create images from text prompts                  |
| Analyze  | Upload an image and ask Gemini to analyze it     |
| Edit     | Upload an image and provide editing instructions |

Image controls include:

* Image mode: `Generation`, `Analysis`, or `Editing`
* Model selection by workflow type
* Temperature and Top-P
* Max output tokens
* Candidate count
* Response mode
* Output MIME type
* Aspect ratio
* Image size where supported
* Google Search grounding where supported
* Google Image Search where supported

## 🎧 Audio

The `Audio` mode supports:

| Workflow       | Description                                  |
| -------------- | -------------------------------------------- |
| Transcribe     | Convert uploaded or recorded audio into text |
| Translate      | Translate uploaded or recorded audio         |
| Text-to-Speech | Generate speech from typed text              |

Audio controls include:

* Task selection
* Model selection
* Language selection
* Voice selection for TTS
* Sample rate
* Output format
* Temperature
* Top-P
* frequency penalty
* presence penalty
* loop and autoplay
* start and end time
* max output tokens
* system prompt template support

## 🔢 Embeddings

The `Embedding` mode supports:

* Embedding model selection
* Encoding format selection
* Dimension selection
* Chunk-size control
* Chunk-overlap control
* Text normalization
* Text chunking
* Embedding generation
* Token, word, unique-word, type-token ratio, and character metrics
* Vector display in a Streamlit data editor

## 📓 Document Q&A

The `Document Q&A` mode supports local document upload and retrieval-augmented question answering.

Supported document loading includes:

* `pdf`
* `txt`
* `md`
* `docx`

The workflow includes:

1. Upload a document.
2. Preview the document when possible.
3. Extract text from the file.
4. Normalize and chunk the text.
5. Generate embeddings using `sentence-transformers`.
6. Store vectors in `sqlite-vec` when available.
7. Fall back to in-memory cosine similarity when vector-table support is unavailable.
8. Retrieve relevant chunks.
9. Build a document-grounded prompt.
10. Return an answer through the chat pipeline.

## 📚 Files API

The `Files` mode exposes Gemini file-oriented workflows through the `Files` wrapper.

Common workflows include:

* File upload
* File ID tracking
* File metadata review
* File purpose/type management
* File-backed prompt workflows where supported by the selected model

## 📦 File Search Stores

The `File Search Stores` mode supports file-search store management through the `FileSearch`
wrapper.

Supported workflows include:

| Workflow | Description                                  |
| -------- | -------------------------------------------- |
| Create   | Create a new file-search store               |
| Retrieve | Retrieve store metadata                      |
| Delete   | Delete a selected file-search store          |
| Upload   | Upload supported files to the selected store |

Supported upload types include:

* `pdf`
* `txt`
* `md`
* `docx`
* `png`
* `jpg`
* `jpeg`

## 🧊 Google Cloud Buckets

The `Google Cloud Buckets` mode supports cloud bucket management through the `CloudBuckets`
wrapper.

Supported workflows include:

| Workflow | Description                                                        |
| -------- | ------------------------------------------------------------------ |
| Create   | Create a new cloud bucket                                          |
| Retrieve | Retrieve cloud bucket metadata                                     |
| Delete   | Delete a selected cloud bucket                                     |
| Upload   | Upload supported files through the available wrapper upload method |

## 📝 Prompt Engineering

The `Prompt Engineering` mode manages reusable prompts stored in the local SQLite `Prompts`
table.

Prompt records include:

| Field       | Description                                |
| ----------- | ------------------------------------------ |
| `PromptsId` | Primary key                                |
| `Caption`   | Display caption used by template selectors |
| `Name`      | Prompt name                                |
| `Text`      | Prompt body                                |
| `Version`   | Prompt version                             |
| `ID`        | External or user-defined identifier        |

Prompt Engineering supports:

* Prompt search
* Prompt sorting
* Prompt pagination
* Prompt selection
* Prompt editing
* Prompt insertion
* Prompt update
* Prompt deletion
* Cascading selected prompts into system instructions

## 📭 Data Export

The `Data Export` mode supports application export workflows, including prompt/system-instruction
export and local data export where configured.

## 🏛️ Data Management

The `Data Management` mode provides a SQLite administration and exploration interface.

Tabs include:

| Tab       | Purpose                                                  |
| --------- | -------------------------------------------------------- |
| Import    | Import external data into SQLite                         |
| Browse    | Browse local SQLite tables                               |
| CRUD      | Insert, update, and delete rows                          |
| Explore   | Preview and inspect current data                         |
| Filter    | Apply advanced filters                                   |
| Aggregate | Run aggregation operations                               |
| Visualize | Build charts from table data                             |
| Admin     | Drop tables, create tables, create indexes, alter schema |
| SQL       | Execute safe read-only SQL queries                       |

Visualization options include:

* Histogram
* Bar chart
* Line chart
* Scatter plot
* Box plot
* Pie chart
* Correlation heatmap

SQL execution is guarded by a read-only validator that allows `SELECT`, `WITH`, `EXPLAIN`, and
read-oriented `PRAGMA` statements while blocking destructive operations such as `INSERT`, `UPDATE`,
`DELETE`, `DROP`, `ALTER`, `CREATE`, `ATTACH`, `DETACH`, `VACUUM`, `REPLACE`, and `TRIGGER`.

## 🧩 Design and Architecture

Jeni uses a traditional layered Streamlit architecture:

| Layer             | Description                                                                                                                                              |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| UI Layer          | Streamlit sidebar, expanders, tabs, chat messages, uploaders, data editors, and charts                                                                   |
| Mode Layer        | Mode-specific Streamlit blocks for Text, Images, Audio, Embedding, Document Q&A, Files, Stores, Buckets, Prompt Engineering, Export, and Data Management |
| Wrapper Layer     | Gemini helper classes imported from `gemini.py`                                                                                                          |
| Persistence Layer | SQLite database under `stores/sqlite`                                                                                                                    |
| Retrieval Layer   | `sentence-transformers`, `sqlite-vec`, chunking, and cosine similarity fallback                                                                          |
| Utility Layer     | Token counting, file saving, text normalization, markdown/XML conversion, usage tracking, and error handling                                             |

## 💻 Capabilities

| Capability           | Description                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------- |
| Text Generation      | Gemini-powered chat and prompt response generation                                                      |
| Google Grounding     | Optional Google Search grounding in Text and supported Image workflows                                  |
| URL Context          | URL inputs can be added to text-generation context                                                      |
| System Prompts       | System-instruction text areas with template loading and XML/Markdown conversion                         |
| Image Generation     | Prompt-to-image generation through Gemini image models                                                  |
| Image Analysis       | Uploaded image analysis using Gemini multimodal models                                                  |
| Image Editing        | Uploaded image editing with text instructions                                                           |
| Audio Transcription  | Uploaded or recorded audio converted to text                                                            |
| Audio Translation    | Uploaded or recorded audio translated into the selected language                                        |
| Text-to-Speech       | Text converted into generated audio                                                                     |
| Embeddings           | Text chunking and vector generation                                                                     |
| Document Q&A         | Local retrieval-augmented document question answering                                                   |
| Files API            | Gemini file upload and metadata workflows                                                               |
| File Search Stores   | File-search store creation, retrieval, deletion, and upload                                             |
| Google Cloud Buckets | Cloud bucket creation, retrieval, deletion, and upload                                                  |
| Prompt Engineering   | SQLite-backed reusable prompt management                                                                |
| Data Export          | Export workflows for prompt and local data assets                                                       |
| Data Management      | SQLite import, browse, CRUD, profile, filter, aggregate, visualize, administer, and SQL query workflows |

## 📁 File Organization

| File / Folder                                                                              | Description                                                                    |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| [`app.py`](https://github.com/is-leeroy-jenkins/Jeni/blob/main/app.py)                     | Main Streamlit application                                                     |
| [`gemini.py`](https://github.com/is-leeroy-jenkins/Jeni/blob/main/gemini.py)               | Gemini wrapper classes                                                         |
| [`config.py`](https://github.com/is-leeroy-jenkins/Jeni/blob/main/config.py)               | Constants, paths, model lists, API defaults, UI labels, and help text          |
| [`requirements.txt`](https://github.com/is-leeroy-jenkins/Jeni/blob/main/requirements.txt) | Python package requirements                                                    |
| `stores/sqlite/Data.db`                                                                    | Local SQLite database for prompts, chat history, embeddings, and imported data |
| `resources/images`                                                                         | Project images, logos, and README assets                                       |
| `resources/setup`                                                                          | API key and setup documentation                                                |

## 🧪 Example Usage

### Text Generation

```python
from gemini import Chat

chat = Chat()
response = chat.generate_text(
    prompt="Explain how random forests reduce overfitting.",
    model="gemini-2.0-flash"
)

print(response)
```

### Embeddings

```python
from gemini import Embeddings

embedding = Embeddings()
vectors = embedding.create(
    text=["Federal budget execution requires accurate obligations tracking."],
    model="text-embedding-004",
    task_type="RETRIEVAL_DOCUMENT"
)

print(vectors)
```

### Image Generation

```python
from gemini import Images

images = Images()
result = images.generate(
    prompt="A clean technical diagram of a retrieval augmented generation pipeline.",
    model="gemini-2.5-flash-image-preview"
)

print(result)
```

### Audio Transcription

```python
from gemini import Transcription

transcriber = Transcription()
text = transcriber.transcribe(
    "audio/meeting.m4a",
    model="gemini-2.0-flash"
)

print(text)
```

### Text-to-Speech

```python
from gemini import TTS

tts = TTS()
audio_bytes = tts.create_speech(
    "Hello from Jeni.",
    model="gemini-2.5-flash-preview-tts",
    voice="Kore"
)

print(type(audio_bytes))
```
## 🚀 Application Badges

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python\&logoColor=white)](#-requirements)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit\&logoColor=white)](#-running-the-streamlit-application)
[![Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=google\&logoColor=white)](#-api-key-setup)
[![SQLite](https://img.shields.io/badge/SQLite-Data%20Store-003B57?logo=sqlite\&logoColor=white)](#-data-management)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Buckets-4285F4?logo=googlecloud\&logoColor=white)](#-google-cloud-buckets)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#-license)

## 📝 Notes

* Some modes depend on model availability and the options exposed by `gemini.py` and `config.py`.
* Some Google Cloud features require valid project and location settings.
* PDF extraction depends on PyMuPDF availability.
* Vector storage uses `sqlite-vec` when available and falls back to cosine similarity when needed.
* The Streamlit sidebar can override configured API keys for the current session.
* The application stores local prompts and data in SQLite.

## 📝 License

Jeni is published under the [MIT License](https://github.com/is-leeroy-jenkins/Jeni/blob/main/LICENSE).

