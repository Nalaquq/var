# Step 1 -- Install WSL2 (Windows Subsystem for Linux)

WSL2 lets you run a full Linux environment inside Windows.
You need this because the training code, Python tools, and CUDA libraries
all work best on Linux.

---

## Prerequisites

- Windows 10 (version 2004 or later) or Windows 11
- Administrator access on your machine
- At least 8 GB of RAM (16 GB recommended)

---

## 1.1 Open PowerShell as Administrator

1. Press the **Windows key** on your keyboard.
2. Type `PowerShell`.
3. Right-click **Windows PowerShell** and choose **Run as administrator**.
4. Click **Yes** when Windows asks for permission.

---

## 1.2 Install WSL

In the PowerShell window, paste this command and press **Enter**:

```powershell
wsl --install
```

This does three things automatically:
- Enables the WSL feature
- Installs the latest Linux kernel
- Installs **Ubuntu** as your default Linux distribution

When it finishes, **restart your computer**.

---

## 1.3 First launch -- create your Linux user

After the restart, Ubuntu should open automatically. If it does not:
1. Press the **Windows key**.
2. Type `Ubuntu` and press **Enter**.

You will be prompted to create a username and password.

> **Important:** This is your Linux username and password, separate from
> your Windows login. Pick something short and easy to remember.
> When you type your password, nothing will appear on screen -- that is
> normal. Just type it and press Enter.

Example:
```
Enter new UNIX username: manning
New password: ********
Retype new password: ********
```

---

## 1.4 Update your system

Now run these two commands to make sure everything is up to date:

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

`sudo` means "run as administrator." It will ask for the password you
just created.

---

## 1.5 Verify WSL2 (not WSL1)

Back in **PowerShell** (not the Ubuntu window), run:

```powershell
wsl --list --verbose
```

You should see something like:

```
  NAME      STATE    VERSION
* Ubuntu    Running  2
```

The **VERSION** column must say **2**. If it says 1, convert it:

```powershell
wsl --set-version Ubuntu 2
```

---

## 1.6 Quick orientation

You now have two "computers" sharing one screen:

| | Windows | WSL (Ubuntu) |
|---|---|---|
| File explorer | `C:\Users\YourName` | Open from Windows: type `\\wsl$\Ubuntu` in the address bar |
| Terminal | PowerShell / CMD | The Ubuntu window you just opened |
| Where to put code | **Don't** -- use WSL instead | `~/` which means `/home/your-username/` |

> **Rule of thumb:** Always store your project files inside WSL
> (the `/home/...` path), not on the Windows `C:\` drive. Accessing
> Windows files from WSL is slow and can cause permission headaches.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `wsl --install` says "not recognized" | Your Windows version is too old. Update Windows first. |
| Ubuntu window closes immediately | Open PowerShell and run `wsl --install -d Ubuntu` |
| VERSION shows 1 | Run `wsl --set-version Ubuntu 2` in PowerShell |
| "Virtual machine" errors | Open BIOS and enable **Virtualization** (VT-x or AMD-V) |

---

**Next:** [Step 2 -- Install VS Code and connect it to WSL](02-install-vscode.md)
