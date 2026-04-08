# Step 2: Setting Up VS Code for WSL Development

VS Code is a free code editor from Microsoft. It can connect directly to your WSL Linux environment so you edit Linux files with a nice graphical editor.

---

## Install VS Code on Windows

1. Go to [https://code.visualstudio.com/](https://code.visualstudio.com/)
2. Click the big **Download** button (it detects your OS automatically)
3. Run the installer
4. During installation, **check all the boxes**, especially:
   - "Add to PATH"
   - "Register Code as an editor for supported file types"
5. Finish the installation

---

## Install the WSL extension

This is what lets VS Code talk to your Linux environment.

1. Open VS Code
2. Click the **Extensions** icon on the left sidebar (it looks like four squares)
3. In the search bar, type `WSL`
4. Find **"WSL"** by Microsoft (it should be the first result)
5. Click **Install**

---

## Connect VS Code to WSL

### Method 1: From the WSL terminal (recommended)

1. Open your Ubuntu terminal
2. Navigate to your project folder:
   ```bash
   cd ~/projects/var
   ```
3. Type:
   ```bash
   code .
   ```
   The `.` means "open the current folder."

The first time you do this, VS Code will install a small server inside WSL. This takes about a minute. After that, VS Code opens and you'll see **"WSL: Ubuntu"** in the bottom-left corner — that means you're connected.

### Method 2: From VS Code

1. Open VS Code
2. Press `Ctrl+Shift+P` to open the Command Palette
3. Type `WSL: Connect to WSL`
4. Select it and VS Code will reconnect to your Linux environment

---

## Essential extensions to install

Once connected to WSL, install these extensions. They will install inside WSL automatically.

Click the Extensions icon (four squares) and search for each:

| Extension | What it does |
|---|---|
| **Python** (by Microsoft) | Python language support, linting, debugging |
| **Pylance** (by Microsoft) | Smart code completion for Python |
| **Jupyter** (by Microsoft) | Run `.ipynb` notebook files inside VS Code |
| **GitLens** (by GitKraken) | See who changed what in the code and when |

> **Important:** When VS Code asks "Install in WSL: Ubuntu?" click **Yes**. Extensions need to be installed in WSL, not on the Windows side.

---

## Select the Python interpreter

After you've set up your virtual environment (covered in [Step 3](03-python-basics.md)):

1. Press `Ctrl+Shift+P`
2. Type `Python: Select Interpreter`
3. Choose the one that shows `.venv/bin/python`

This tells VS Code to use the project's Python environment (not the system Python).

---

## VS Code basics for beginners

### The interface

```
+------------------------------------------------------------------+
|  Menu Bar (File, Edit, View, etc.)                               |
+--------+---------------------------------------------------------+
|        |                                                         |
| Side   |  Editor Area                                            |
| Bar    |  (where you edit code)                                  |
|        |                                                         |
|  [E]   |                                                         |
|  [S]   |                                                         |
|  [G]   |                                                         |
|  [D]   |                                                         |
|  [X]   |                                                         |
+--------+---------------------------------------------------------+
|  Terminal Panel (toggle with Ctrl+`)                             |
+------------------------------------------------------------------+
```

- **[E] Explorer** — see your files and folders
- **[S] Search** — find text across all files
- **[G] Source Control** — Git integration
- **[D] Debug** — run and debug code
- **[X] Extensions** — install add-ons

### Essential keyboard shortcuts

| Shortcut | What it does |
|---|---|
| `Ctrl+`` ` (backtick) | Toggle the built-in terminal |
| `Ctrl+Shift+P` | Open Command Palette (search for any action) |
| `Ctrl+P` | Quick-open a file by name |
| `Ctrl+S` | Save the current file |
| `Ctrl+Shift+E` | Show the file explorer |
| `Ctrl+Shift+F` | Search across all files |
| `Ctrl+/` | Toggle comment on selected lines |
| `Ctrl+Z` | Undo |
| `Ctrl+Shift+Z` | Redo |

### The integrated terminal

The terminal at the bottom of VS Code is a real Linux terminal — it's the same as your Ubuntu terminal. You can run all the same commands here.

To open it: Press `` Ctrl+` `` (the backtick key, usually above Tab).

> **Tip:** You can have multiple terminals open. Click the **+** icon in the terminal panel to add another one.

---

## Working with files

- **Create a new file:** Right-click in the Explorer panel and select "New File"
- **Open a folder:** File menu > Open Folder (or `code /path/to/folder` from terminal)
- **Save all files:** `Ctrl+K` then `S`

---

## Troubleshooting

**VS Code opens but doesn't connect to WSL:**
- Make sure the WSL extension is installed
- Try: `Ctrl+Shift+P` > "WSL: Connect to WSL"

**`code .` command not found in WSL terminal:**
- Close and reopen your terminal
- If still broken, open VS Code from Windows, press `Ctrl+Shift+P`, type "Shell Command: Install 'code' command in PATH"

**Extensions not working:**
- Make sure extensions are installed in **WSL** (not locally). Look for the "Install in WSL" button.

---

## Next step

With VS Code connected to WSL, move on to [Step 3: Python and Virtual Environments](03-python-basics.md).
