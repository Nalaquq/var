# Step 4: Git and GitHub

Git is a version control system — it tracks every change you make to your code so you can go back in time, collaborate with others, and never lose work. GitHub is a website that hosts your Git projects online.

---

## Install Git

Git usually comes pre-installed in WSL/Ubuntu. Check:

```bash
git --version
```

If not installed:

```bash
sudo apt-get install git
```

---

## Configure Git (one-time setup)

Tell Git who you are. This labels your changes with your name:

```bash
git config --global user.name "Manning Lasso"
git config --global user.email "your-email@example.com"
```

Set the default branch name to `main`:

```bash
git config --global init.defaultBranch main
```

---

## Key concepts

| Concept | What it means |
|---|---|
| **Repository (repo)** | A project folder tracked by Git |
| **Commit** | A snapshot of your code at a point in time (like a save point) |
| **Branch** | A separate line of development (like a copy you can experiment on) |
| **Remote** | A copy of the repo on a server (like GitHub) |
| **Clone** | Download a repo from GitHub to your computer |
| **Push** | Upload your commits to GitHub |
| **Pull** | Download the latest changes from GitHub |

---

## The basic Git workflow

```
 Edit files → Stage changes → Commit → Push to GitHub
```

### 1. Check what's changed

```bash
git status
```

This shows:
- **Red files** — changed but not staged
- **Green files** — staged and ready to commit
- **Untracked files** — new files Git doesn't know about yet

### 2. Stage your changes

"Staging" means selecting which changes to include in your next commit.

```bash
# Stage a specific file
git add src/train.py

# Stage multiple files
git add src/train.py src/models.py

# Stage all changed files (use with caution)
git add .
```

### 3. Commit (save a snapshot)

```bash
git commit -m "Add dropout to BiLSTM model"
```

The `-m` flag lets you write a short message describing what you changed. Good commit messages explain **why**, not just **what**:

- Good: `"Add dropout to reduce overfitting on Dive class"`
- Bad: `"Updated stuff"`
- Bad: `"asdf"`

### 4. Push to GitHub

```bash
git push
```

This uploads your commits to GitHub so your supervisor can see them.

---

## Setting up GitHub

### Create a GitHub account

1. Go to [https://github.com](https://github.com)
2. Click **Sign Up**
3. Follow the steps to create your account

### Authenticate with GitHub from WSL

The easiest way is with GitHub CLI:

```bash
# Install GitHub CLI
sudo apt-get install gh

# Log in
gh auth login
```

Follow the prompts:
1. Choose **GitHub.com**
2. Choose **HTTPS**
3. Choose **Login with a web browser**
4. It gives you a code — open the URL in your browser and paste the code

After this, Git commands like `push` and `pull` will work without asking for your password every time.

---

## Cloning this project

If your supervisor has shared the repo on GitHub:

```bash
cd ~
mkdir -p projects
cd projects
git clone https://github.com/YOUR-ORG/var.git
cd var
```

Replace the URL with the actual repository URL.

---

## Viewing history

```bash
# See recent commits
git log --oneline -10

# See what changed in the last commit
git diff HEAD~1

# See who changed a specific line (blame)
git blame src/train.py
```

---

## Branches (for later)

Branches let you work on something without affecting the main code. You probably won't need this right away, but here's the basic idea:

```bash
# Create a new branch and switch to it
git checkout -b my-experiment

# Do your work, commit as usual...
git add .
git commit -m "Trying larger learning rate"

# Switch back to main
git checkout main

# Merge your experiment into main (if it worked)
git merge my-experiment
```

---

## The .gitignore file

Some files shouldn't be tracked by Git (large data files, temporary files, virtual environments). A `.gitignore` file lists patterns to exclude:

```
# Already ignored in this project:
.venv/              # virtual environment (big, machine-specific)
__pycache__/        # Python cache files
results/            # training outputs (large)
checkpoints/        # model weights (large)
feature_cache/      # cached features (very large)
*.pyc               # compiled Python files
```

If you accidentally commit a large file, ask your supervisor for help rather than trying to fix it yourself — it can get complicated.

---

## Common Git commands cheat sheet

| Command | What it does |
|---|---|
| `git status` | See what's changed |
| `git add filename` | Stage a file for commit |
| `git commit -m "message"` | Commit staged changes |
| `git push` | Upload to GitHub |
| `git pull` | Download latest from GitHub |
| `git log --oneline -10` | See last 10 commits |
| `git diff` | See unstaged changes |
| `git diff --staged` | See staged changes |
| `git clone URL` | Download a repo |

---

## Common errors and fixes

**`fatal: not a git repository`**
You're not inside a Git project folder. `cd` into your project first.

**`error: failed to push some refs`**
Someone else pushed changes. Pull first, then push:
```bash
git pull
git push
```

**`Permission denied (publickey)`**
Run `gh auth login` again to re-authenticate.

**Committed something you shouldn't have?**
Don't panic. Ask your supervisor — there are ways to undo commits safely.

---

## Next step

Move on to [Step 5: NVIDIA Drivers and CUDA](05-cuda-gpu-setup.md) to set up GPU support.
