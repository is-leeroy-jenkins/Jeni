# Getting Started

This guide assumes a Windows development workstation using PowerShell from the Jeni repository root.

## Prerequisites

| Requirement           | Purpose                                                          |
|-----------------------|------------------------------------------------------------------|
| Python 3.10+          | Runtime for the Streamlit application and documentation tooling. |
| Git                   | Clone and manage the repository.                                 |
| PowerShell            | Run the Windows setup and build commands.                        |
| Gemini API key        | Required for Gemini-backed workflows.                            |
| Google Cloud settings | Required only for cloud bucket workflows.                        |
| MkDocs dependencies   | Required to build and publish this documentation site.           |

## Clone the Repository

```powershell
git clone https://github.com/is-leeroy-jenkins/Jeni.git
cd Jeni
```

## Create and Activate a Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once for the current user and then activate the environment again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

## Install Application Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Install Documentation Dependencies

```powershell
pip install -r requirements-docs.txt
```

## Configure API Keys

For local development, set the required keys through the Streamlit sidebar or through PowerShell environment variables.

```powershell
setx GEMINI_API_KEY "your-gemini-api-key"
setx GOOGLE_API_KEY "your-google-api-key"
setx GOOGLE_CSE_ID "your-google-custom-search-id"
setx GOOGLEMAPS_API_KEY "your-google-maps-api-key"
setx GOOGLE_CLOUD_PROJECT_ID "your-google-cloud-project-id"
setx GOOGLE_CLOUD_LOCATION "us-central1"
```

Close and reopen PowerShell after using `setx`, then reactivate the virtual environment.

## Run the Streamlit Application

```powershell
streamlit run app.py
```

The application should open at:

```text
http://localhost:8501
```

## Serve the Documentation Locally

```powershell
mkdocs serve
```

The documentation preview should open at:

```text
http://127.0.0.1:8000/Jeni/
```

## Build the Documentation Locally

```powershell
mkdocs build --strict
```

A successful build creates a local `site/` folder containing static HTML, CSS, JavaScript, search assets, and sitemap files.

## Publish the Documentation to GitHub Pages

```powershell
mkdocs build --strict
mkdocs gh-deploy --force
```

Then configure GitHub Pages:

```text
Source: Deploy from a branch
Branch: gh-pages
Folder: / (root)
```

The published site should resolve to:

```text
https://is-leeroy-jenkins.github.io/Jeni/
```

## Troubleshooting

| Symptom                                           | Likely Cause                                                                          | Correction                                                                                                    |
|---------------------------------------------------|---------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `mkdocs` is not recognized                        | Documentation dependencies are not installed in the active environment.               | Run `pip install -r requirements-docs.txt` and confirm `mkdocs --version`.                                    |
| API reference does not render                     | `mkdocstrings` cannot import the module.                                              | Install application dependencies first with `pip install -r requirements.txt`.                                |
| `app.py` API page fails if converted to `::: app` | `app.py` performs Streamlit session-state work during import.                         | Keep the app page narrative-only until runtime code is moved behind `main()` and `if __name__ == "__main__"`. |
| GitHub Pages opens the README                     | Pages is pointed to the source branch or `gh-pages` does not contain generated files. | Run `mkdocs gh-deploy --force` and set Pages to `gh-pages` branch `/ root`.                                   |
| `custom_dir` error                                | `mkdocs.yml` points to a missing `docs/overrides` directory.                          | Do not use `custom_dir` unless override templates actually exist.                                             |
