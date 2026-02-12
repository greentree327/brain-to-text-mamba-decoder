# Git Workflow Reference

Quick reference for version control operations.

## Initial Setup

```bash
git init
git add .
git commit -m "Initial commit: Modularized Mamba+GRU decoder (7th place solution)"
git remote add origin https://github.com/greentree327/brain-to-text-mamba-decoder.git
git branch -M main
git push -u origin main
```

## Common Operations

```bash
# Check status
git status

# View changes
git diff

# Stage and commit
git add .
git commit -m "Update: description"

# Push to remote
git push origin main
```

## Merge Conflicts

If the remote has diverged:

```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```
