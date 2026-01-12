import os
import json
from pathlib import Path

def validate_json_folder(root_dir="../raw_data/"):
    print(f"--- Starting JSON Integrity Check in: {root_dir} ---")
    
    # Find all JSON files in the directory and subdirectories
    json_files = list(Path(root_dir).rglob("*.json"))
    total_files = len(json_files)
    corrupted_files = []

    print(f"Found {total_files} JSON files. Scanning...")

    for i, file_path in enumerate(json_files):
        # Feedback every 100 files
        if i % 100 == 0:
            print(f"Progress: {i}/{total_files} files checked...")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"\nCORRUPTED FILE FOUND: {file_path}")
            print(f"Error: {e}")
            corrupted_files.append(str(file_path))

    print("\n--- SCAN COMPLETE ---")
    if corrupted_files:
        print(f"Found {len(corrupted_files)} corrupted files.")
        
        # Saving the list of corrupted files to a log
        with open("corrupted_json_list.txt", "w") as log:
            for f in corrupted_files:
                log.write(f + "\n")
        print("The list of corrupted files has been saved in 'corrupted_json_list.txt'.")
        
    else:
        print("All JSON files are valid!")

if __name__ == "__main__":
    #raw_data_path = "your path to raw data" 
    #validate_json_folder(raw_data_path)
    validate_json_folder()