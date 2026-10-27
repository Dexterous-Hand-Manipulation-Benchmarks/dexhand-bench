import os


def scan_all_configs():
    base_dir = "."
    print(f"[INFO] Searching entire project root for yaml/json/pt files: {base_dir}\n")

    for root, dirs, files in os.walk(base_dir):
        # Skip virtualenvs and git directories to save time
        if ".git" in root or "venv" in root or "conda" in root:
            continue

        for file in files:
            if file.endswith((".yaml", ".yml", ".json", ".pt")):
                full_p = os.path.join(root, file)
                print(f"Found: {full_p}")
                # If it's a small yaml or json, peek inside for clip
                if file.endswith((".yaml", ".yml", ".json")) and os.path.getsize(full_p) < 10000:
                    try:
                        with open(full_p, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            if "clip" in content.lower() or "box" in content.lower():
                                print(f"  -> MATCH CONTENT IN {file}")
                                for line in content.splitlines():
                                    if "clip" in line.lower() or "env_name" in line.lower():
                                        print(f"     {line.strip()}")
                    except Exception:
                        pass


if __name__ == "__main__":
    scan_all_configs()