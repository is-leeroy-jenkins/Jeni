# User Guide

Jeni is operated through a Streamlit interface. Launch it from the repository root with:

```powershell
streamlit run app.py
```

Use the sidebar to select the active mode, provider model, runtime parameters, optional system instructions, and mode-specific controls.

## Python Usage Examples

The examples below show how to use Jeni's Gemini wrapper classes directly from Python. These
examples assume they are run from the repository root or from an environment where the Jeni source
directory is on `PYTHONPATH`.

```python
import config as cfg
from gemini import (
    Chat,
    Images,
    Embeddings,
    TTS,
    Transcription,
    Translation,
    Files,
    FileSearch,
    CloudBuckets,
)
```

Most examples require a valid Gemini API key configured through `config.py` or environment
variables.

```powershell
setx GEMINI_API_KEY "your-gemini-api-key"
setx GOOGLE_API_KEY "your-google-api-key"
```

Restart PowerShell after using `setx`.



### Chat

The `Chat` class supports text-generation workflows, including standard prompts, system
instructions, grounding tools, URL context, and file-search-oriented requests where supported by the
selected model.

#### Basic text generation

```python
from gemini import Chat

chat = Chat()

response = chat.generate_text(
    prompt="Explain the difference between budget authority, obligations, and outlays.",
    model="gemini-2.5-flash",
    temperature=0.2,
    max_tokens=1000,
)

print(response)
```

#### Text generation with system instructions

```python
from gemini import Chat

chat = Chat()

response = chat.generate_text(
    prompt="Summarize the risks of operating under a continuing resolution.",
    model="gemini-2.5-flash",
    temperature=0.2,
    max_tokens=1200,
    instruct=(
        "You are a federal budget analyst. Use precise appropriations terminology, "
        "distinguish budget authority from obligations and outlays, and answer in "
        "a concise executive briefing style."
    ),
)

print(response)
```

#### Build URL context before a request

```python
from gemini import Chat

chat = Chat()

urls = chat.build_urls(
    urls=[
        "https://www.whitehouse.gov/omb/",
        "https://home.treasury.gov/",
    ],
    max_urls=2,
)

response = chat.generate_text(
    prompt="Using the provided references, summarize the role of OMB and Treasury in federal budget execution.",
    model="gemini-2.5-flash",
    temperature=0.2,
    max_tokens=1200,
    urls=urls,
)

print(response)
```

#### Check model-supported tools

```python
from gemini import Chat

chat = Chat()

tools = chat.get_supported_tools("gemini-2.5-flash")

print(tools)
```



### Images

The `Images` class supports image generation, image analysis, and image editing workflows.

#### Generate an image

```python
from pathlib import Path
from gemini import Images

images = Images()

image = images.generate(
    prompt=(
        "Create a dark-themed software architecture diagram for a Streamlit application "
        "that uses Gemini wrappers, SQLite persistence, document retrieval, and Google Cloud services."
    ),
    model="gemini-2.5-flash-image",
    aspect="16:9",
    resolution="1K",
    response_modalities="image",
)

if image is not None:
    output_path = Path("docs/images/jeni-generated-architecture.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved image to {output_path}")
```

#### Analyze an image

```python
from gemini import Images

images = Images()

analysis = images.analyze(
    prompt=(
        "Review this architecture diagram for documentation quality. Identify missing labels, "
        "unclear relationships, and improvements for a technical audience."
    ),
    path="docs/images/jeni-architecture.png",
    model="gemini-2.5-flash-image",
    response_modalities="text",
)

print(analysis)
```

#### Edit an image

```python
from pathlib import Path
from gemini import Images

images = Images()

edited = images.edit(
    prompt=(
        "Improve text contrast, preserve the dark theme, make the layer boundaries clearer, "
        "and keep the diagram suitable for MkDocs documentation."
    ),
    path="docs/images/jeni-architecture.png",
    model="gemini-2.5-flash-image",
    aspect="16:9",
    resolution="1K",
    response_modalities="image",
)

if edited is not None:
    output_path = Path("docs/images/jeni-architecture-edited.png")
    edited.save(output_path)
    print(f"Saved edited image to {output_path}")
```



### Embeddings

The `Embeddings` class creates vector embeddings from text input.

#### Create one embedding

```python
from gemini import Embeddings

embeddings = Embeddings()

vector = embeddings.create(
    text="Budget authority allows agencies to incur obligations and make payments from the Treasury.",
    model="gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT",
    title="Budget Authority Definition",
)

print(type(vector))
print(len(vector))
print(vector[:10])
```

#### Create embeddings for multiple text chunks

```python
from gemini import Embeddings

embeddings = Embeddings()

chunks = [
    "Budget authority is authority provided by law to incur financial obligations.",
    "Obligations are binding agreements that will result in outlays.",
    "Outlays are payments made to liquidate obligations.",
]

vectors = embeddings.create(
    text=chunks,
    model="gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT",
    title="Federal Budget Terms",
)

print(f"Created {len(vectors)} vectors.")
print(f"First vector length: {len(vectors[0])}")
```



### TTS

The `TTS` class converts text into spoken audio.

#### Generate WAV bytes

```python
from gemini import TTS

tts = TTS()

audio_bytes = tts.create_speech(
    text="Jeni is ready to support text, image, audio, document, and data workflows.",
    model="gemini-2.5-flash-preview-tts",
    voice="Kore",
    format="audio/wav",
)

if audio_bytes is not None:
    print(f"Generated {len(audio_bytes)} bytes.")
```

#### Generate and save a WAV file

```python
from gemini import TTS

tts = TTS()

output_path = tts.create_speech(
    text="This is a generated audio briefing from Jeni.",
    filepath="outputs/jeni-briefing.wav",
    model="gemini-2.5-flash-preview-tts",
    voice="Kore",
    format="audio/wav",
    instruct="Read in a calm, professional briefing tone.",
)

print(output_path)
```



### Transcription

The `Transcription` class transcribes local audio files.

#### Transcribe an audio file

```python
from gemini import Transcription

transcription = Transcription()

text = transcription.transcribe(
    path="samples/audio/meeting.wav",
    model="gemini-3-flash-preview",
    language="English",
    mime_type="audio/wav",
    temperature=0.2,
)

print(text)
```

#### Transcribe a specific time range

```python
from gemini import Transcription

transcription = Transcription()

text = transcription.transcribe(
    path="samples/audio/meeting.wav",
    model="gemini-3-flash-preview",
    language="English",
    mime_type="audio/wav",
    start_time=30.0,
    end_time=90.0,
)

print(text)
```



### Translation

The `Translation` class translates spoken audio into a target language.

#### Translate audio to English

```python
from gemini import Translation

translation = Translation()

translated_text = translation.translate(
    path="samples/audio/spanish-briefing.wav",
    model="gemini-3-flash-preview",
    language="English",
    source="Spanish",
    mime_type="audio/wav",
)

print(translated_text)
```

#### Translate a selected time range

```python
from gemini import Translation

translation = Translation()

translated_text = translation.translate(
    path="samples/audio/french-briefing.wav",
    model="gemini-3-flash-preview",
    language="English",
    source="French",
    mime_type="audio/wav",
    start_time=10.0,
    end_time=60.0,
)

print(translated_text)
```



### Files

The `Files` class supports Gemini file upload, file retrieval, file listing, document summarization,
document search, and file-backed workflows.

#### Upload a file

```python
from gemini import Files

files = Files()

uploaded_file = files.upload(
    filepath="samples/documents/policy.pdf",
    name="Policy Reference",
)

print(uploaded_file)
```

#### Retrieve a file

```python
from gemini import Files

files = Files()

file_metadata = files.retrieve(
    file_id="files/example-file-id"
)

print(file_metadata)
```

#### Summarize a document

```python
from gemini import Files

files = Files()

summary = files.summarize(
    prompt=(
        "Summarize this document for an executive audience. Include purpose, key requirements, "
        "deadlines, risks, and recommended follow-up actions."
    ),
    filepath="samples/documents/policy.pdf",
    model="gemini-2.0-flash",
    temperature=0.2,
    max_tokens=1500,
)

print(summary)
```

#### Search a document

```python
from gemini import Files

files = Files()

answer = files.search(
    prompt="Find every section that discusses auditability, internal controls, or reporting requirements.",
    filepath="samples/documents/policy.pdf",
    model="gemini-2.0-flash",
    temperature=0.2,
    max_tokens=1500,
)

print(answer)
```

#### List configured cloud-backed files

```python
from gemini import Files

files = Files()

available_files = files.list()

for file_name in available_files:
    print(file_name)
```



### FileSearch

The `FileSearch` class manages Gemini File Search Store resources.

#### List file-search stores

```python
from gemini import FileSearch

stores = FileSearch()

available_stores = stores.list()

for store in available_stores:
    print(store)
```

#### Create a file-search store

```python
from gemini import FileSearch

stores = FileSearch()

store = stores.create(
    name="Budget Execution Reference Library"
)

print(store)
```

#### Retrieve a file-search store

```python
from gemini import FileSearch

stores = FileSearch()

store = stores.retrieve(
    store_id="fileSearchStores/example-store-id"
)

print(store)
```

#### Delete a file-search store

```python
from gemini import FileSearch

stores = FileSearch()

deleted = stores.delete(
    store_id="fileSearchStores/example-store-id",
    force=True,
)

print(deleted)
```



### CloudBuckets

The `CloudBuckets` class wraps Google Cloud Storage bucket workflows.

#### List bucket contents

```python
from gemini import CloudBuckets

buckets = CloudBuckets()

objects = buckets.list(
    bucket="jeni-financial"
)

for obj in objects:
    print(obj.name)
```

#### Upload a file to a bucket

```python
from gemini import CloudBuckets

buckets = CloudBuckets()

blob = buckets.upload(
    path="samples/documents/policy.pdf",
    bucket="jeni-financial",
    name="regulations/policy.pdf",
)

print(blob.name)
```

#### Retrieve a bucket object

```python
from gemini import CloudBuckets

buckets = CloudBuckets()

blob = buckets.retrieve(
    bucket="jeni-financial",
    name="regulations/policy.pdf",
)

print(blob)
```

#### Delete a bucket object

```python
from gemini import CloudBuckets

buckets = CloudBuckets()

deleted = buckets.delete(
    bucket="jeni-financial",
    name="regulations/policy.pdf",
)

print(deleted)
```



### End-to-End Example: Document Summary to Executive Brief

This example uploads a document, summarizes it, and then uses the Chat wrapper to produce a polished
executive briefing.

```python
from gemini import Files, Chat

files = Files()
chat = Chat()

summary = files.summarize(
    prompt=(
        "Summarize this document. Focus on purpose, requirements, deadlines, risks, "
        "and recommended actions."
    ),
    filepath="samples/documents/policy.pdf",
    model="gemini-2.0-flash",
    temperature=0.2,
    max_tokens=1500,
)

briefing = chat.generate_text(
    prompt=(
        "Convert the following document summary into an executive briefing with "
        "Background, Key Findings, Risks, and Recommended Actions sections.\n\n"
        f"{summary}"
    ),
    model="gemini-2.5-flash",
    temperature=0.2,
    max_tokens=1500,
)

print(briefing)
```



### End-to-End Example: Document Retrieval Preparation

This example creates embeddings for document chunks so they can be stored or compared in a retrieval
workflow.

```python
from gemini import Embeddings

chunks = [
    "The program office must validate obligations before quarterly reporting.",
    "The finance office must reconcile obligations against outlays each month.",
    "The audit team must review internal controls and document corrective actions.",
]

embeddings = Embeddings()

vectors = embeddings.create(
    text=chunks,
    model="gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT",
    title="Program Controls Reference",
)

for index, vector in enumerate(vectors, start=1):
    print(f"Chunk {index}: {len(vector)} dimensions")
```



### End-to-End Example: Architecture Image Analysis

This example analyzes the Jeni architecture image stored in the documentation assets folder.

```python
from gemini import Images

images = Images()

analysis = images.analyze(
    prompt=(
        "Review this architecture image for technical documentation. Identify whether the "
        "main layers, service boundaries, request flow, configuration flow, and persistence "
        "flow are clear."
    ),
    path="docs/images/jeni-architecture.png",
    model="gemini-2.5-flash-image",
    response_modalities="text",
)

print(analysis)
```

## Usage Examples

The examples below show common Jeni workflows by mode and by the wrapper class that supports each
workflow. They are written from the user-interface perspective first, with the corresponding
implementation class noted where it helps connect the application behavior to the API documentation.

### Text Mode

**Backing class:** `gemini.Chat`

Use Text mode when you want a Gemini-backed response to a prompt, with optional system instructions,
model settings, grounding tools, URL context, or file-search support.

#### Example: Generate a budget-analysis summary

1. Select **Text** mode from the sidebar.

2. Select a supported Gemini text model.

3. Enter system instructions such as:

   ```text
   You are a federal budget analyst. Answer with concise, source-aware analysis.
   ```

4. Enter a prompt such as:

   ```text
   Summarize the major budget-execution risks associated with delayed apportionment,
   continuing resolutions, and late-year obligation pressure.
   ```

5. Adjust generation settings if needed:

    * lower temperature for formal analysis;
    * higher maximum tokens for longer explanations;
    * Google Search grounding when current external context is required.

6. Submit the prompt and review the generated response.

#### Example: Ask a grounded question with URL context

1. Select **Text** mode.

2. Enable URL-context support if the selected model supports it.

3. Add one or more reference URLs in the URL input area.

4. Ask a focused question, for example:

   ```text
   Based on the provided references, identify the key compliance requirements and
   summarize them as an implementation checklist.
   ```

5. Review the answer and any returned grounding metadata.



### Images Mode

**Backing class:** `gemini.Images`

Use Images mode for image generation, image analysis, and image editing workflows.

#### Example: Generate an architecture graphic

1. Select **Images** mode.

2. Open the **Generate** tab.

3. Select an image-capable Gemini model.

4. Enter a detailed prompt, for example:

   ```text
   Create a dark-themed technical architecture diagram for a Streamlit application
   that uses Gemini wrappers, local SQLite persistence, document retrieval, and
   Google Cloud integrations.
   ```

5. Choose an aspect ratio such as `16:9`.

6. Choose an output MIME type such as `image/png`.

7. Submit the request.

8. Save the generated image into `docs/images/` if it will be used in the documentation.

#### Example: Analyze an uploaded screenshot

1. Select **Images** mode.

2. Open the **Analyze** tab.

3. Upload an image or screenshot.

4. Enter a prompt such as:

   ```text
   Review this screenshot for layout issues, missing labels, visual hierarchy problems,
   and documentation-readiness.
   ```

5. Submit the analysis request.

6. Use the response to refine the UI, README, or documentation page.

#### Example: Edit an uploaded image

1. Select **Images** mode.

2. Open the **Edit** tab.

3. Upload the source image.

4. Enter edit instructions such as:

   ```text
   Make the diagram wider, improve text contrast, preserve the dark theme, and make
   the architecture layers easier to distinguish.
   ```

5. Submit the request.

6. Review the returned edited image before replacing any documentation asset.



### Audio Mode

Audio mode supports transcription, translation, and text-to-speech. These workflows are implemented
through the `gemini.Transcription`, `gemini.Translation`, and `gemini.TTS` classes.

#### Example: Transcribe an uploaded audio file

**Backing class:** `gemini.Transcription`

1. Select **Audio** mode.
2. Choose the **Transcribe** workflow.
3. Upload a supported audio file, such as WAV, MP3, AIFF, AAC, OGG, or FLAC.
4. Select the expected language or leave it on automatic detection.
5. Optionally provide a start time and end time to transcribe only part of the file.
6. Submit the transcription request.
7. Review the returned transcript and copy it into notes, documentation, or downstream analysis.

#### Example: Translate spoken audio

**Backing class:** `gemini.Translation`

1. Select **Audio** mode.
2. Choose the **Translate** workflow.
3. Upload the audio file.
4. Select the target language.
5. Optionally identify the source language.
6. Submit the translation request.
7. Review the translated text output.

#### Example: Generate spoken audio from text

**Backing class:** `gemini.TTS`

1. Select **Audio** mode.

2. Choose the **Text-to-Speech** workflow.

3. Enter the text to convert into speech.

4. Select a TTS-capable model.

5. Select a voice.

6. Optionally adjust speaking style through instructions, such as:

   ```text
   Read this in a calm, professional briefing tone.
   ```

7. Generate the audio.

8. Play or download the generated WAV output.



### Embedding Mode

**Backing class:** `gemini.Embeddings`

Use Embedding mode to convert text into vector representations for semantic comparison, retrieval,
clustering, classification, or document Q&A support.

#### Example: Create an embedding for semantic search

1. Select **Embedding** mode.

2. Choose an embedding model.

3. Paste source text, for example:

   ```text
   The Working Capital Fund supports shared services, technology modernization,
   and reimbursable service delivery across participating offices.
   ```

4. Select the task type when appropriate, such as:

    * `RETRIEVAL_DOCUMENT` for indexed source material;
    * `RETRIEVAL_QUERY` for search questions;
    * `SEMANTIC_SIMILARITY` for comparison workflows.

5. Configure dimensions if the model supports dimensionality controls.

6. Generate the embedding.

7. Review the vector output and any text metrics shown in the interface.

#### Example: Embed multiple chunks

1. Select **Embedding** mode.
2. Paste multiple paragraphs or chunked text.
3. Configure chunk size and overlap where available.
4. Generate embeddings.
5. Use the output for retrieval, similarity scoring, or local vector storage.



### Document Q&A Mode

**Backing classes:** `gemini.Files`, `gemini.Embeddings`, and local retrieval helpers

Use Document Q&A mode when you want Jeni to answer questions using uploaded documents as context.

#### Example: Ask a question against a PDF

1. Select **Document Q&A** mode.

2. Upload one or more PDF files.

3. Preview the extracted text when available.

4. Build or rebuild the retrieval index.

5. Ask a document-grounded question, for example:

   ```text
   What are the main reporting requirements in this document, and what offices are
   responsible for each requirement?
   ```

6. Review the answer and supporting context.

7. Refine the question if the answer needs more precision.

#### Example: Compare requirements across uploaded documents

1. Upload multiple policy, budget, or technical documents.

2. Build the retrieval index.

3. Ask:

   ```text
   Compare the requirements across these documents and identify overlapping,
   conflicting, or missing implementation steps.
   ```

4. Review the response for document-grounded similarities and differences.



### Files Mode

**Backing class:** `gemini.Files`

Use Files mode for Gemini file upload, file metadata, document summarization, document search, and
file-backed prompt workflows.

#### Example: Upload a file for later use

1. Select **Files** mode.
2. Choose the upload workflow.
3. Select a local file.
4. Provide a display name when required.
5. Upload the file.
6. Record the returned file metadata or file identifier if it will be reused in another workflow.

#### Example: Summarize an uploaded document

1. Select **Files** mode.

2. Upload or select a document.

3. Enter a summary prompt such as:

   ```text
   Summarize this document for a senior executive. Focus on purpose, major decisions,
   risks, deadlines, and required follow-up actions.
   ```

4. Submit the request.

5. Review the generated summary.

#### Example: Search within an uploaded file

1. Select **Files** mode.

2. Upload or select the document to search.

3. Enter a targeted question, for example:

   ```text
   Find every section that discusses auditability, internal controls, or reporting
   requirements.
   ```

4. Submit the search request.

5. Review the response and supporting references.



### File Search Stores Mode

**Backing class:** `gemini.FileSearch`

Use File Search Stores mode to create, retrieve, list, delete, and manage Gemini file-search stores.

#### Example: Create a file-search store

1. Select **File Search Stores** mode.

2. Choose the create-store workflow.

3. Enter a display name, such as:

   ```text
   Budget Execution Reference Library
   ```

4. Create the store.

5. Confirm that the new store appears in the available store list.

#### Example: Retrieve a file-search store

1. Select **File Search Stores** mode.
2. Choose a store from the available collection list.
3. Retrieve the store metadata.
4. Confirm the display name, resource name, and available status information.

#### Example: Delete a file-search store

1. Select **File Search Stores** mode.
2. Select the store to delete.
3. Confirm that the selected store is not needed by an active workflow.
4. Delete the store.
5. Refresh the store list.



### Google Cloud Buckets Mode

**Backing class:** `gemini.CloudBuckets`

Use Google Cloud Buckets mode when Jeni needs to interact with Google Cloud Storage buckets and
objects.

#### Example: Upload a document to a bucket

1. Select **Google Cloud Buckets** mode.
2. Choose the upload workflow.
3. Select a local file.
4. Enter the target bucket name.
5. Enter an object name or accept the default object name derived from the file path.
6. Upload the file.
7. Confirm that the object appears in the bucket listing.

#### Example: Retrieve a bucket object

1. Select **Google Cloud Buckets** mode.
2. Enter the bucket name.
3. Enter the object name.
4. Retrieve the object metadata.
5. Use the returned information to confirm that the expected cloud object is available.

#### Example: List bucket contents

1. Select **Google Cloud Buckets** mode.
2. Enter the bucket name.
3. Run the list workflow.
4. Review the returned object names and identifiers.
5. Use the results to select files for downstream document, data, or retrieval workflows.



### Prompt Engineering Mode

Prompt Engineering mode manages reusable prompts stored in the local SQLite `Prompts` table.

#### Example: Create a reusable system instruction

1. Select **Prompt Engineering** mode.

2. Create a new prompt record.

3. Enter a name such as:

   ```text
   Federal Budget Analyst
   ```

4. Enter reusable instruction text, such as:

   ```text
   You are a federal budget analyst. Use precise budget terminology, distinguish
   budget authority from obligations and outlays, and present conclusions in a
   concise executive format.
   ```

5. Save the prompt record.

6. Reuse the prompt in Text, Document Q&A, or file-backed workflows.

#### Example: Update an existing prompt

1. Select **Prompt Engineering** mode.
2. Search or browse existing prompt records.
3. Select the prompt to edit.
4. Update the name, caption, text, version, or identifier.
5. Save the updated record.
6. Confirm that the revised prompt appears in the prompt list.



### Data Management Mode

Data Management mode supports SQLite-backed data import, browsing, filtering, aggregation,
visualization, administration, and guarded SQL queries.

#### Example: Import a CSV file

1. Select **Data Management** mode.
2. Open the **Import** tab.
3. Upload a CSV file.
4. Confirm the inferred table name and column structure.
5. Import the data into SQLite.
6. Open the **Browse** or **Explore** tab to confirm the imported rows.

#### Example: Filter and aggregate imported data

1. Select **Data Management** mode.
2. Open the **Filter** tab.
3. Choose the imported table.
4. Apply one or more column filters.
5. Open the **Aggregate** tab.
6. Select grouping columns and numeric measures.
7. Run the aggregation.
8. Review the summarized output.

#### Example: Run a guarded read-only SQL query

1. Select **Data Management** mode.

2. Open the **SQL** tab.

3. Enter a read-only query, such as:

   ```sql
   SELECT
       AccountName,
       SUM(BY) AS BudgetYearAmount
   FROM CombinedSchedules
   GROUP BY AccountName
   ORDER BY BudgetYearAmount DESC;
   ```

4. Run the query.

5. Review the result table.

6. Avoid destructive SQL operations in user-facing workflows.



### Data Export Mode

Data Export mode supports exporting local data assets, prompt records, generated artifacts, or
application-managed datasets where configured.

#### Example: Export prompt records

1. Select **Data Export** mode.
2. Choose the prompt-record export workflow.
3. Select the desired output format.
4. Generate the export.
5. Save the exported file for backup, migration, or documentation review.

#### Example: Export imported data

1. Select **Data Export** mode.
2. Choose the table or dataset to export.
3. Select the output format.
4. Generate the export.
5. Confirm that the exported file contains the expected rows and columns.



### Common End-to-End Workflow

The following workflow combines several Jeni modes:

1. Use **Files** mode to upload a policy or budget document.
2. Use **Document Q&A** mode to ask targeted questions against the uploaded document.
3. Use **Embedding** mode to inspect or generate vectors for semantic retrieval.
4. Use **Prompt Engineering** mode to save a reusable analytical instruction.
5. Use **Text** mode to generate a final executive summary.
6. Use **Data Export** mode to export reusable records or supporting data.

This pattern is useful when turning source documents into structured analysis, reusable prompts,
documentation content, or decision-support outputs.
