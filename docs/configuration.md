# Configuration

Jeni reads configuration from `config.py`, operating-system environment variables, and Streamlit session state. Values entered in the Streamlit sidebar are used for the active session, while environment variables and `config.py` provide application defaults.

## Provider and Cloud Settings

| Variable                  | Used For                                            | Required                                         |
|---------------------------|-----------------------------------------------------|--------------------------------------------------|
| `GEMINI_API_KEY`          | Gemini API access.                                  | Required for Gemini workflows.                   |
| `GOOGLE_API_KEY`          | Google API access and Gemini fallback key behavior. | Recommended.                                     |
| `GOOGLE_CSE_ID`           | Google Custom Search integration.                   | Required for CSE-backed search workflows.        |
| `GOOGLEMAPS_API_KEY`      | Google Maps-related workflows.                      | Required only for Maps-enabled workflows.        |
| `GOOGLE_CLOUD_PROJECT_ID` | Google Cloud project routing.                       | Required for cloud bucket workflows.             |
| `GOOGLE_CLOUD_LOCATION`   | Google Cloud region or location.                    | Required for cloud workflows that need a region. |
| `GEOCODING_API_KEY`       | Geocoding-related workflows.                        | Optional unless geocoding is used.               |

## Local Paths

| Setting           | Purpose                                                               |
|-------------------|-----------------------------------------------------------------------|
| `ROOT_DIR`        | Repository root resolved from `config.py`.                            |
| `DOCS_DIR`        | Local documentation directory.                                        |
| `LOG_DIR`         | Logging directory, defaulting to `logging`.                           |
| `LOG_PATH`        | Exception logging database path.                                      |
| `LOG_FILE`        | Exception logging table name.                                         |
| `DB_PATH`         | SQLite application database path: `stores/sqlite/datamodels/Data.db`. |
| `FAVICON`         | Streamlit favicon path.                                               |
| `LOGO_PATH`       | Jeni logo path.                                                       |
| `AUDIO_TEST_FILE` | Default audio test file.                                              |

## Application Modes

Jeni exposes the following modes through the Gemini configuration layer:

| Mode                 | Backing Wrapper                                 |
|----------------------|-------------------------------------------------|
| Text                 | `Chat`                                          |
| Images               | `Images`                                        |
| Audio                | `TTS`, `Translation`, `Transcription`           |
| Embedding            | `Embeddings`                                    |
| Document Q&A         | `Files` plus local document retrieval helpers   |
| Files                | `Files`                                         |
| File Search Stores   | `FileSearch`                                    |
| Google Cloud Buckets | `CloudBuckets`                                  |
| Prompt Engineering   | SQLite `Prompts` table workflows                |
| Data Management      | SQLite data-management workflows                |
| Data Export          | Export helpers for local data and prompt assets |

## Documentation Settings

The documentation site is configured in `mkdocs.yml`.

| Setting      | Value                                                 |
|--------------|-------------------------------------------------------|
| `site_name`  | `Jeni Documentation`                                  |
| `site_url`   | `https://is-leeroy-jenkins.github.io/Jeni/`           |
| `repo_url`   | `https://github.com/is-leeroy-jenkins/Jeni`           |
| `repo_name`  | `is-leeroy-jenkins/Jeni`                              |
| Theme        | Material for MkDocs, dark-first, blue accent styling. |
| API renderer | `mkdocstrings[python]` using Google-style docstrings. |

## Security Guidance

Never commit API keys, cloud credentials, local secrets, `.env` files, generated databases containing sensitive content, or private document uploads. Use environment variables, Streamlit secrets, GitHub repository secrets, or local-only configuration instead.

For GitHub Actions publishing, store secrets in GitHub only when the workflow must run authenticated tests or cloud checks. The documentation workflow included here does not need provider credentials to build the static site if API reference imports do not execute provider calls at import time.
