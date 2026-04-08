# Step 3: Python and Virtual Environments

This guide covers installing Python in WSL and understanding virtual environments — isolated spaces where each project keeps its own set of packages without interfering with other projects.

---

## Check if Python is already installed

Open your WSL terminal and run:

```bash
python3 --version
```

You should see something like `Python 3.10.12` or higher. If you get "command not found," install it:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv
```

---

## What is Python?

Python is a programming language commonly used for data science and machine learning. A few things to know:

- Python files end in `.py`
- You run them with `python3 my_script.py`
- Python uses **indentation** (spaces) to structure code — this matters!
- Comments start with `#`

### Quick example

```python
# This is a comment
name = "Manning"
print(f"Hello, {name}!")

# A simple loop
for i in range(5):
    print(i)  # prints 0, 1, 2, 3, 4
```

---

## What is pip?

`pip` is Python's package manager. It downloads and installs libraries (pre-written code) that other people have made. For example:

```bash
pip install numpy        # math library
pip install pandas       # data tables
pip install torch        # deep learning (PyTorch)
```

---

## What is a virtual environment?

A virtual environment is an **isolated copy of Python** for one specific project.

### Why do we need this?

Imagine:
- Project A needs `numpy version 1.21`
- Project B needs `numpy version 1.24`

If you install both into the same Python, they'll conflict. A virtual environment gives each project its own separate set of packages.

### How it works

```
System Python (shared by everything)
    |
    +-- Project A's .venv/  (has numpy 1.21, torch 2.0)
    |
    +-- Project B's .venv/  (has numpy 1.24, tensorflow 2.13)
```

Each `.venv` folder is self-contained. Deleting it and recreating it is safe.

---

## Creating and using a virtual environment

### Create it (one time)

```bash
cd ~/projects/var/local_training
python3 -m venv .venv
```

This creates a `.venv` folder in your project directory.

### Activate it (every time you open a terminal)

```bash
source .venv/bin/activate
```

You'll see `(.venv)` appear at the start of your terminal prompt:

```
(.venv) manning@laptop:~/projects/var/local_training$
```

This means the virtual environment is active. Any `pip install` commands now install into `.venv`, not your system Python.

### Deactivate it (when you're done)

```bash
deactivate
```

> **Important:** You need to activate the virtual environment every time you open a new terminal. If you see errors like "ModuleNotFoundError: No module named 'torch'," you probably forgot to activate.

---

## For this project: use setup.sh

You don't need to manually create the virtual environment for MVFoul — the `setup.sh` script does it all for you:

```bash
cd ~/projects/var/local_training
bash setup.sh
```

This script:
1. Creates the `.venv` virtual environment
2. Detects your CUDA version
3. Installs PyTorch with GPU support
4. Installs all other required packages
5. Verifies everything works

After running it, activate with:

```bash
source .venv/bin/activate
```

---

## Useful Python/pip commands

| Command | What it does |
|---|---|
| `python3 --version` | Check Python version |
| `pip list` | Show all installed packages |
| `pip install package_name` | Install a package |
| `pip install --upgrade package_name` | Update a package |
| `pip freeze` | List packages with exact versions |
| `which python3` | Show which Python executable is being used |

### Verify your environment is correct

After activating your virtual environment, run:

```bash
which python3
```

It should show something like:
```
/home/manning/projects/var/local_training/.venv/bin/python3
```

If it shows `/usr/bin/python3` instead, you're using the system Python — activate the virtual environment first.

---

## Running Python scripts

```bash
# Make sure you're in the right directory and venv is active
cd ~/projects/var/local_training
source .venv/bin/activate

# Run a script
python3 src/train.py --approach B --config src/config.yaml --debug
```

### What the flags mean

- `python3 src/train.py` — run the file `train.py` inside the `src` folder
- `--approach B` — a command-line argument telling the script which model to train
- `--config src/config.yaml` — points to the configuration file
- `--debug` — runs on a small subset of data for quick testing

---

## Common errors and fixes

**`ModuleNotFoundError: No module named 'torch'`**
You forgot to activate the virtual environment:
```bash
source .venv/bin/activate
```

**`pip: command not found`**
Install pip:
```bash
sudo apt-get install python3-pip
```

**`Permission denied` when running pip:**
Never use `sudo pip install`. Always use a virtual environment instead.

**`python: command not found` (but `python3` works):**
Ubuntu ships with `python3`, not `python`. Always type `python3`.

---

## Next step

Now that you understand Python and virtual environments, move on to [Step 4: Git and GitHub](04-git-github.md).
