def load_metadata_from_txt(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            metadata = f.read()
        print(metadata)
    except Exception as e:
        print(f"❌ Error reading metadata: {e}")
        return ""   
    
load_metadata_from_txt("../metadata/djia_metadata.txt")