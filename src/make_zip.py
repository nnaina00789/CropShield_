import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZIP_OUT = ROOT / "CropShield_Project.zip"

INCLUDE_DIRS = [
    "app",
    "src",
    "checkpoints",
    "demo_images",
    "reports",
    "splits"
]

INCLUDE_FILES = [
    "README.md",
    "requirements.txt",
    ".gitignore",
    "CropShield_Presentation.pptx",
    "CropShield_Project.ipynb",
    "CropShield_Viva_and_Project_Guide.pdf"
]

EXCLUDE_PATTERNS = [
    "__pycache__",
    ".git",
    ".venv",
    ".DS_Store",
    "*.pyc"
]

print(f"Creating project ZIP archive at: {ZIP_OUT}")
with zipfile.ZipFile(ZIP_OUT, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Add explicit root files
    for fname in INCLUDE_FILES:
        fpath = ROOT / fname
        if fpath.exists():
            zipf.write(fpath, arcname=f"CropShield/{fname}")
            print(f"Added file: {fname}")

    # Add directories
    for dname in INCLUDE_DIRS:
        dpath = ROOT / dname
        if not dpath.exists():
            continue
        for file in dpath.rglob("*"):
            if file.is_file():
                # Check exclusions
                if any(part in file.parts for part in ["__pycache__", ".git", ".venv", ".pytest_cache"]):
                    continue
                if file.suffix in [".pyc", ".pyo"]:
                    continue
                
                rel_path = file.relative_to(ROOT)
                arc_name = f"CropShield/{rel_path.as_posix()}"
                zipf.write(file, arcname=arc_name)

print("ZIP creation complete!")
size_mb = ZIP_OUT.stat().st_size / (1024 * 1024)
print(f"Archive Size: {size_mb:.2f} MB")
