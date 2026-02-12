# Push Repository to GitHub

## Step 1: Create GitHub Repository
1. Go to https://github.com/new
2. Name: `brain-to-text-mamba-decoder`
3. **Do NOT** initialize with README (you already have one)
4. Click "Create repository"

## Step 2: Initialize Git (if not already)
```powershell
cd c:\Users\user\Desktop\brain-to-text-mamba-decoder
git init
```

## Step 3: Add All Files
```powershell
git add .
```

## Step 4: Commit
```powershell
git commit -m "Initial commit: Modularized Mamba+GRU decoder (7th place solution)"
```

## Step 5: Add Remote
Replace `YOUR_USERNAME` with your GitHub username:
```powershell
git remote add origin https://github.com/YOUR_USERNAME/brain-to-text-mamba-decoder.git
```

## Step 6: Push
```powershell
git branch -M main
git push -u origin main
```

**Note:** If you get a "rejected" error because GitHub initialized with files, run:
```powershell
git pull origin main --allow-unrelated-histories
git push -u origin main
```

## Step 7: Update Colab Notebook
After pushing, update [notebooks/inference_colab.ipynb](notebooks/inference_colab.ipynb) Cell 3 (Clone Repository):
```python
!git clone https://github.com/YOUR_USERNAME/brain-to-text-mamba-decoder.git
```

Replace `YOUR_USERNAME` with your actual GitHub username (e.g., `greentree327`).

---

## Quick Reference

### Check status
```powershell
git status
```

### View unsaved changes
```powershell
git diff
```

### Save credentials (optional)
```powershell
git config --global credential.helper store
```
