# Step 7: Using Google Colab

This guide explains how to use Google Colab to run the project notebooks, including how to upload and download files.

---

## What is Google Colab?

Google Colab is a free online environment that lets you run Python notebooks in the cloud. It provides a GPU so you can train models without needing your own graphics card. Think of it as a temporary computer in the cloud that comes pre-installed with Python and machine learning tools.

---

## Opening a notebook in Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File > Open notebook**
3. Select the **GitHub** tab or the **Upload** tab:
   - **GitHub:** Paste the repository URL and select the notebook
   - **Upload:** Download the `.ipynb` file from the repo and upload it manually

---

## Uploading files to Colab

Colab runs on a temporary virtual machine — it does **not** have access to files on your computer. If a notebook needs files (like CSVs from a previous step), you must upload them.

### Method 1: Upload via the file browser (easiest)

1. In Colab, click the **folder icon** in the left sidebar to open the file browser
2. Click the **upload icon** (page with an up arrow) at the top of the file browser
3. Select the files from your computer
4. They will appear in `/content/` — this is Colab's working directory

> **Important:** Uploaded files are **temporary**. They are deleted when your Colab session ends (after ~12 hours of inactivity, or when you disconnect). Always download any outputs you need before closing Colab.

### Method 2: Upload with Python code

You can also upload files using a code cell. Add a cell at the top of the notebook and run:

```python
from google.colab import files
uploaded = files.upload()
```

A file picker dialog will appear. Select your files and they will be uploaded to the current working directory (`/content/`).

### Method 3: Mount Google Drive (best for large or reusable files)

If you want your files to persist between sessions, use Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

This will ask you to sign in to your Google account. Once mounted, your Drive files are at `/content/drive/MyDrive/`. You can create a project folder there:

```python
import shutil
# Copy a file from Drive to the working directory
shutil.copy('/content/drive/MyDrive/MVFoul/class_weights.csv', '/content/')
```

---

## Downloading files from Colab

When a notebook produces output files (CSVs, plots, model checkpoints), you need to download them before your session ends.

### Method 1: Download via the file browser

1. Click the **folder icon** in the left sidebar
2. Find the file you want
3. Click the **three dots** next to the file name
4. Select **Download**

### Method 2: Download with Python code

Add a code cell and run:

```python
from google.colab import files
files.download('filename.csv')
```

Your browser will download the file to your local Downloads folder.

### Method 3: Save to Google Drive

If you mounted Drive (see above), copy files there:

```python
import shutil
shutil.copy('/content/filename.csv', '/content/drive/MyDrive/MVFoul/')
```

---

## Uploading CSVs from Phase 0 into Phase 1

After running `01_dataset_statistics.ipynb`, you will have downloaded CSV files to your computer (stored in `notebooks/outputs/` in the repo). When you open `02_three_baselines.ipynb`, you need to upload these files:

1. Open `02_three_baselines.ipynb` in Colab
2. Click the **folder icon** in the left sidebar
3. Click the **upload icon** and select these files from `notebooks/outputs/`:
   - `class_weights.csv`
   - `dive_crosstab.csv`
   - `mvfoul_all_actions.csv`
   - `mvfoul_dives_only.csv`
4. Verify they uploaded by running in a code cell:
   ```python
   import os
   os.listdir('/content/')
   ```
5. You should see your CSV files listed in the output

---

## Tips for working in Colab

- **Save your work:** Colab autosaves to your Google Drive, but you can also do **File > Save a copy in Drive** to be safe
- **Check your GPU:** Run `!nvidia-smi` in a code cell to confirm a GPU is assigned. If it says "No GPU", go to **Runtime > Change runtime type** and select **T4 GPU**
- **Session timeouts:** Free Colab sessions disconnect after ~90 minutes of inactivity or ~12 hours total. Download your outputs before you leave
- **Restart runtime:** If something breaks, go to **Runtime > Restart runtime** to start fresh (your uploaded files will still be there, but any variables in memory are cleared)
- **RAM limits:** Free Colab gives ~12 GB of RAM. If you get a "session crashed" message, your notebook used too much memory. Try enabling `DEBUG_MODE = True` to use a smaller dataset

---

## Quick reference

| Task | How to do it |
|---|---|
| Upload files | Folder icon > upload icon, or `files.upload()` |
| Download files | Folder icon > three dots > Download, or `files.download()` |
| Check GPU | Run `!nvidia-smi` in a cell |
| Change GPU | Runtime > Change runtime type > T4 GPU |
| Check working directory | Run `!ls /content/` in a cell |
| Persistent storage | Mount Google Drive with `drive.mount()` |
