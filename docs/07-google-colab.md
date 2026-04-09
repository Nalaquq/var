# Step 7: Using Google Colab

This guide walks you through running the project notebooks in Google Colab. No installation on your computer is needed — everything runs in the cloud.

---

## What is Google Colab?

Google Colab is a free website that lets you run Python code in the cloud. It gives you a temporary computer with a GPU (graphics card) so you can train models without needing your own. Think of it like Google Docs, but for code.

**Important things to know:**
- Your files are **temporary**. When you close the tab or the session times out (~90 min idle, ~12 hours total), everything is deleted.
- Always **download your results** before closing Colab.

---

## How to run the unified notebook

The project has one main notebook that does everything: `notebooks/00_unified_pipeline.ipynb`. It downloads the data, shows you statistics, and trains all three models — all in one place.

### Step 1: Open the notebook in Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File > Open notebook**
3. Choose one of these options:
   - **Upload tab:** Download `00_unified_pipeline.ipynb` from the repo to your computer first, then upload it here
   - **GitHub tab:** Paste the repository URL and select the notebook

### Step 2: Turn on the GPU

The training parts of the notebook need a GPU. Free Colab gives you one:

1. Click **Runtime > Change runtime type** (in the menu bar at the top)
2. Under "Hardware accelerator", select **T4 GPU**
3. Click **Save**

To verify the GPU is working, run this in any code cell:
```
!nvidia-smi
```
You should see a table showing the GPU name (like "Tesla T4"). If it says "No GPU", repeat the steps above.

### Step 3: Run the notebook

You have two options:

- **Run everything at once:** Click **Runtime > Run all**. This takes ~15 minutes in debug mode.
- **Run one cell at a time:** Click each cell and press **Shift+Enter** (or click the play button on the left side of the cell). Go from top to bottom — don't skip cells.

The notebook has 9 parts:
1. Install packages
2. Download data
3. Dataset statistics and plots
4. Prepare data for training
5. Train Approach A (MViT video model)
6. Train Approach B (BiLSTM pose model)
7. Train Approach C (ST-GCN skeleton model)
8. Compare all three approaches
9. Download your results

### Step 4: Download your results

When the notebook finishes, run the last cell (Part 9). It will automatically download the CSV files and training results to your computer's Downloads folder.

You can also download files manually:
1. Click the **folder icon** in the left sidebar to open the file browser
2. Navigate to the file you want
3. Click the **three dots** next to the file name
4. Select **Download**

**Save these files into `notebooks/outputs/` in your local copy of the repo.**

---

## Uploading and downloading files (general)

### Uploading files to Colab

If you ever need to upload a file (for example, a CSV or a config file):

1. Click the **folder icon** in the left sidebar
2. Click the **upload icon** (page with an up arrow) at the top
3. Select files from your computer
4. They appear in `/content/` — Colab's working directory

Or use Python code in a cell:
```python
from google.colab import files
uploaded = files.upload()
```

### Downloading files from Colab

To download a file with Python code:
```python
from google.colab import files
files.download('filename.csv')
```

### Using Google Drive for persistent storage

If you want files to survive between sessions, mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

Then copy files to/from Drive:
```python
import shutil
# Save a result to Drive
shutil.copy('/content/results/comparison_table.csv', '/content/drive/MyDrive/')
# Load a file from Drive
shutil.copy('/content/drive/MyDrive/my_file.csv', '/content/')
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No GPU" or training is very slow | **Runtime > Change runtime type > T4 GPU** |
| "Session crashed" or out of memory | Set `DEBUG_MODE = True` in the config cell and restart: **Runtime > Restart runtime** |
| Session disconnected | Your session timed out. Re-open the notebook, turn on the GPU, and run all cells again. Uploaded files are gone — but the notebook re-downloads the data automatically. |
| `ModuleNotFoundError` | Run the install cell (Part 1) first. If it still fails, try **Runtime > Restart runtime** then run from the top. |
| Plots look wrong or variables not found | Make sure you ran cells in order from top to bottom. Don't skip cells. |

---

## Quick reference

| Task | How to do it |
|------|-------------|
| Open a notebook | File > Open notebook > Upload or GitHub tab |
| Turn on GPU | Runtime > Change runtime type > T4 GPU |
| Run all cells | Runtime > Run all |
| Run one cell | Click the cell, then Shift+Enter |
| Upload files | Folder icon > upload icon |
| Download files | Folder icon > three dots > Download |
| Check GPU | Run `!nvidia-smi` in a cell |
| Restart if stuck | Runtime > Restart runtime |
| Check what files exist | Run `!ls /content/` in a cell |
| Persistent storage | Mount Google Drive (see above) |

---

## About the original two-notebook workflow

The repo also contains the original separate notebooks:
- `01_dataset_statistics.ipynb` — statistics only
- `02_three_baselines.ipynb` — training only

These are kept for reference but are **not recommended** for new users. The unified notebook (`00_unified_pipeline.ipynb`) combines both into a single file so you don't need to transfer data between them.
