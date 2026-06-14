Developers Guide - For running app1.py locally (pulling the model files via git LFS) and contribiting.

# ML GitHub Workflow (with Git LFS)

Everytime you download or pull the models pickle or whatever the model files, you have to start from git lfs install, git lfs track ,  git add .gitattributes, git commit -m "Setup Git LFS tracking", git push origin master



## 0. One-time setup (first time only)

### Install LFS

```bash
git lfs install
```

### Track large ML files (run once per repo)

```bash
git lfs track "*.h5"
git lfs track "*.hdf5"
git lfs track "*.pickle"
git lfs track "*.pck"
git lfs track "*.npz"
git lfs track "*.joblib"

```

### Commit LFS config

```bash
git add .gitattributes
git commit -m "Setup Git LFS tracking"
```

git push origin master

**Why:**
This tells Git which files should be stored in LFS instead of normal Git (avoids repo bloat).

---

# 🚀 1. Everyday workflow (coding / editing files)

## Step 1: Make changes

Edit:

* `.py`
* `README.md`
* Streamlit app
* configs

---

## Step 2: Stage changes

```bash
git add .
```

**Why:**
Prepares changes to be saved in Git.

---

## Step 3: Commit changes

```bash
git commit -m "update model pipeline"
```

**Why:**
Creates a snapshot of your project.

---

## Step 4: Push to GitHub

```bash
git push origin master
```

**Why:**
Uploads your commits to GitHub.

---

# 📦 2. Adding ML models / large files

If you add new `.h5`, `.pickle`, etc:

### Step 1 (only if new type)

```bash
git lfs track "*.h5"
```

### Step 2

```bash
git add .
git commit -m "add new model"
git push origin master
```

**Why LFS:**
Stores large files outside Git history → keeps repo light.

---

# 📥 3. Cloning a ML + LFS repo

## Step 1: Clone repo

```bash
git clone https://github.com/username/repo.git
cd repo
```

---

## Step 2: Enable LFS

```bash
git lfs install
git lfs pull
```

**Why:**
Git clones only small pointer files — LFS pull downloads actual model weights.

---

## Step 3 (important check)

```bash
git lfs ls-files
```

You should see model files listed.

---

# 🔄 4. Pulling updates from GitHub

```bash
git pull origin master
git lfs pull
```

**Why two steps:**

* `git pull` → updates code
* `git lfs pull` → downloads updated models

---

# ⚠️ 5. Common mistakes (avoid these)

❌ Don’t do LFS setup every time
❌ Don’t forget `git lfs pull` after cloning
❌ Don’t manually commit large `.h5` without LFS tracking
❌ Don’t ignore `data/` if it contains models (unless intentional)

---

# 🧠 Simple mental model

| Action        | Tool                    |
| ------------- | ----------------------- |
| Code changes  | Git                     |
| Large models  | Git LFS                 |
| Download repo | git clone               |
| Get updates   | git pull + git lfs pull |

---

# 🧪 One-line workflow summary

### Everyday dev:

```bash
git add .
git commit -m "msg"
git push
```

### After clone:

```bash
git clone <repo>
git lfs install
git lfs pull
```

