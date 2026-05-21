# WSL2 + Conda Development Workflow

Recommended setup for this repository:

1. Keep the working copy inside the WSL filesystem, not under `/mnt/c` or
   `/mnt/d`, for much faster Git and Python file operations.

   ```bash
   mkdir -p ~/code
   cd ~/code
   git clone https://github.com/devilpusher/vqe-r12.git
   cd vqe-r12
   ```

2. Create the conda environment from the repository file.

   ```bash
   conda env create -f environment.yml
   conda activate vqecodex
   ```

   If the environment already exists:

   ```bash
   conda env update -n vqecodex -f environment.yml --prune
   conda activate vqecodex
   ```

3. Run a quick syntax check before committing changes.

   ```bash
   make check
   ```

4. Run the scripts from the repository root so default input/output paths match.

   ```bash
   make step1
   make step2
   make step3
   make step4b
   make step5a
   make step5b
   make step5c
   ```

5. Keep generated scientific outputs local. The repository ignores `.npz`,
   `.npy`, Psi4 `.out`, summary `.txt`, and comparison `.csv` files by default.

## Daily Loop

```bash
cd ~/code/vqe-r12
conda activate vqecodex
git pull
make check
# edit/run scripts
git status
git add <changed-files>
git commit -m "Describe the change"
git push
```

## Editor

Use VS Code with the Remote - WSL extension if you want a GUI editor while still
running Python, Git, Psi4, and conda inside Ubuntu.

