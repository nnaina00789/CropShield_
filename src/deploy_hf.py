import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

SPACE_README_HEADER = """---
title: CropShield
emoji: 🌿
colorFrom: green
colorTo: emerald
sdk: streamlit
app_file: app.py
pinned: false
---
"""

def prepare_bundle(bundle_dir: Path, readme_body: str):
    import shutil

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    (bundle_dir / "README.md").write_text(SPACE_README_HEADER + "\n" + readme_body, encoding="utf-8")
    shutil.copy("requirements.txt", bundle_dir / "requirements.txt")
    shutil.copy("app/app.py", bundle_dir / "app.py")
    (bundle_dir / "app").mkdir(exist_ok=True)
    shutil.copy("app/remedies.json", bundle_dir / "app/remedies.json")
    shutil.copy("app/remedies_hi.json", bundle_dir / "app/remedies_hi.json")
    shutil.copytree("src", bundle_dir / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (bundle_dir / "splits").mkdir(exist_ok=True)
    shutil.copy("splits/train.csv", bundle_dir / "splits/train.csv")
    (bundle_dir / "checkpoints").mkdir(exist_ok=True)
    shutil.copy("checkpoints/leaf_detector.pt", bundle_dir / "checkpoints/leaf_detector.pt")
    ckpt = "checkpoints/resnet18_field.pt" if Path("checkpoints/resnet18_field.pt").exists() else "checkpoints/resnet18.pt"
    shutil.copy(ckpt, bundle_dir / "checkpoints/resnet18_field.pt")
    shutil.copytree("demo_images", bundle_dir / "demo_images")
    print(f"Prepared Hugging Face Space bundle at {bundle_dir}")


def main():
    parser = argparse.ArgumentParser(description="Deploy CropShield to Hugging Face Spaces")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face Write Token")
    parser.add_argument("--space", default=None, help="Space repo_id, e.g. username/CropShield")
    parser.add_argument("--prepare-only", action="store_true", help="Only build the local deploy_space/ bundle")
    args = parser.parse_args()

    readme_body = Path("README.md").read_text(encoding="utf-8") if Path("README.md").exists() else "# CropShield"
    bundle_dir = Path("deploy_space")
    prepare_bundle(bundle_dir, readme_body)

    if args.prepare_only:
        return

    api = HfApi(token=args.token)
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = args.space or f"{username}/CropShield"

    print(f"Creating / updating Hugging Face Space: {repo_id} ...")
    create_repo(repo_id=repo_id, token=args.token, repo_type="space", space_sdk="streamlit", exist_ok=True)
    upload_folder(
        repo_id=repo_id,
        folder_path=str(bundle_dir),
        repo_type="space",
        token=args.token,
        commit_message="Deploy CropShield Streamlit app with LeafDetector & field-tuned ResNet18",
    )
    print(f"\nSuccessfully deployed! Public URL: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
