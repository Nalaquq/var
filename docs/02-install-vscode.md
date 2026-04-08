# Step 2 -- Install VS Code and Connect It to WSL

VS Code is the code editor you will use for this project. It has a special
feature that lets you edit files inside WSL as if they were local files.

---

## 2.1 Download and install VS Code

1. Go to <https://code.visualstudio.com/> in your **Windows** browser.
2. Click the big blue **Download for Windows** button.
3. Run the installer. Accept all defaults.
   - Make sure **"Add to PATH"** is checked (it usually is by default).

---

## 2.2 Install the WSL extension

1. Open VS Code (on Windows).
2. Press **Ctrl+Shift+X** to open the Extensions panel.
3. In the search bar, type `WSL`.
4. Find **WSL** by Microsoft (it has a green icon) and click **Install**.

This extension lets VS Code run its backend inside your WSL Ubuntu
so you can edit Linux files seamlessly.

---

## 2.3 Connect VS Code to WSL

**Option A -- from the Ubuntu terminal (recommended):**

1. Open your Ubuntu terminal.
2. Navigate to any folder (or just stay in your home directory).
3. Type:

```bash
code .
```

The first time you run this, it will install the VS Code Server inside
WSL. Wait for it to finish. VS Code will open on Windows, but the
bottom-left corner will show a green badge that says **WSL: Ubuntu**.

**Option B -- from inside VS Code:**

1. Press **Ctrl+Shift+P** to open the Command Palette.
2. Type `WSL: Connect to WSL` and press **Enter**.

---

## 2.4 Install recommended extensions

Once connected to WSL, install these extensions (they install inside WSL):

Press **Ctrl+Shift+X** and search for each one:

| Extension | Why you need it |
|---|---|
| **Python** (by Microsoft) | Syntax highlighting, linting, debugger |
| **Pylance** (by Microsoft) | Smarter autocomplete for Python |
| **Jupyter** (by Microsoft) | View and run `.ipynb` notebooks |

> **Tip:** When you search for extensions while connected to WSL, you may
> see a button that says **"Install in WSL: Ubuntu"** instead of just
> "Install." That is correct -- click it.

---

## 2.5 Open a folder in WSL

1. In VS Code (connected to WSL), press **Ctrl+K** then **Ctrl+O**
   (or go to File > Open Folder).
2. Navigate to the project folder. For example:
   ```
   /home/manning/var
   ```
3. Click **OK**.

You are now editing files that live inside Linux, using a Windows editor.
Everything -- the terminal, the file tree, the Python interpreter -- runs
inside WSL.

---

## 2.6 Use the built-in terminal

Press **Ctrl+`** (the backtick key, above Tab) to open a terminal inside
VS Code. This terminal runs inside WSL automatically when you are
connected. You will use this terminal for all commands in later steps.

---

## 2.7 Verify everything works

In the VS Code terminal (Ctrl+`), run:

```bash
uname -a
```

You should see something containing `Linux` and `microsoft`. This
confirms the terminal is running inside WSL, not Windows.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `code .` says "command not found" | Close and reopen your Ubuntu terminal, then try again |
| No green "WSL: Ubuntu" badge | Click the green icon in the bottom-left and select "Connect to WSL" |
| Extensions say "Install Locally" | Make sure you are connected to WSL first, then install them |

---

**Next:** [Step 3 -- Install Python in WSL](03-install-python.md)
