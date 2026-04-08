# Step 1: Installing WSL2 (Windows Subsystem for Linux)

WSL2 lets you run a full Linux environment inside Windows. We need this because most machine-learning tools work best on Linux.

---

## What is WSL?

Think of it as a Linux computer running inside your Windows computer. You get a real Linux terminal (command line) without needing to dual-boot or use a virtual machine. Your Windows files and Linux files can see each other.

---

## Prerequisites

- Windows 10 (version 2004 or later) or Windows 11
- An internet connection
- Administrator access on your machine

---

## Step-by-step installation

### 1. Open PowerShell as Administrator

- Click the **Start** menu
- Type `PowerShell`
- Right-click **Windows PowerShell** and choose **Run as administrator**
- Click **Yes** when prompted

### 2. Install WSL

In the PowerShell window, type this command and press Enter:

```powershell
wsl --install
```

This does three things automatically:
- Enables the WSL feature
- Installs the latest Linux kernel
- Installs **Ubuntu** as your default Linux distribution

> **Note:** If you already have WSL installed, you can run `wsl --install -d Ubuntu` to make sure you have Ubuntu.

### 3. Restart your computer

After the install finishes, **restart your PC**. This is required.

### 4. Set up your Linux username and password

After restarting, Ubuntu will launch automatically (or find it in your Start menu). It will ask you to create:

- **Username**: Pick something simple, all lowercase, no spaces (e.g., `manning`)
- **Password**: You'll type it and **nothing will appear on screen** — that's normal in Linux. Just type it and press Enter.

> **Important:** Remember this password! You'll need it when installing software (anytime you use `sudo`).

### 5. Update your system

Run these two commands (copy-paste each line, press Enter):

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

`sudo` means "run as administrator." It will ask for the password you just created.

---

## Verify it's working

Run this command:

```bash
wsl.exe --list --verbose
```

You should see something like:

```
  NAME      STATE           VERSION
* Ubuntu    Running         2
```

The **VERSION** column must say **2**. If it says 1, run:

```powershell
wsl --set-version Ubuntu 2
```

---

## Key concepts you'll need

| Concept | What it means |
|---|---|
| **Terminal** | The text-based window where you type commands |
| **`sudo`** | "Super user do" — runs a command as administrator |
| **`apt-get`** | Ubuntu's package manager (like an app store, but for the terminal) |
| **Home directory** | Your personal folder, located at `/home/your-username/` |
| **`~`** (tilde) | Shortcut that means "my home directory" |

---

## Navigating your file system

Your Windows files live at `/mnt/c/` inside Linux. For example:
- `C:\Users\Manning\Desktop` is `/mnt/c/Users/Manning/Desktop` in WSL

Your Linux files live at `\\wsl$\Ubuntu\home\your-username\` when viewed from Windows.

> **Best practice:** Keep project files on the Linux side (`~/projects/`) for better performance. Accessing Windows files (`/mnt/c/...`) from Linux is slower.

---

## Common commands cheat sheet

| Command | What it does | Example |
|---|---|---|
| `ls` | List files in current folder | `ls` |
| `cd` | Change directory (folder) | `cd ~/projects` |
| `pwd` | Print current directory | `pwd` |
| `mkdir` | Make a new directory | `mkdir my-folder` |
| `cp` | Copy a file | `cp file.txt backup.txt` |
| `mv` | Move or rename a file | `mv old.txt new.txt` |
| `rm` | Delete a file (careful!) | `rm unwanted.txt` |
| `cat` | Show contents of a file | `cat readme.txt` |
| `clear` | Clear the terminal screen | `clear` |

---

## Troubleshooting

**"WSL is not installed" error:**
Open PowerShell as admin and run:
```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```
Then restart and try again.

**"The virtual machine could not be started" error:**
You may need to enable virtualization in your BIOS. Restart your PC, enter BIOS (usually by pressing F2, F12, or Delete during startup), and enable "Intel VT-x" or "AMD-V."

**Forgot your Linux password:**
Open PowerShell and run:
```powershell
wsl -u root passwd your-username
```

---

## Next step

Once WSL2 is installed and working, move on to [Step 2: Setting Up VS Code](02-vscode-setup.md).
