import os
import shutil
from tqdm import tqdm
import zipfile


def delete_log_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith("log_") and file.endswith(".txt"):
                file_path = os.path.join(root, file)
                print(f"Deleting file: {file_path}")
                # input()
                os.remove(file_path)
            if file.startswith("evolution.json"):
                file_path = os.path.join(root, file)
                print(f"Deleting file: {file_path}")
                os.remove(file_path)


def zip_directory(directory):
    parent_dir = os.path.basename(directory)
    zip_file = os.path.join(directory, "zipped.zip")
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as archive:
        total_files = sum(len(files) for _, _, files in os.walk(directory))
        with tqdm(total=total_files, desc="Zipping directory") as pbar:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(".zip"):
                        continue
                    file_path = os.path.join(root, file)
                    arcname = os.path.join(parent_dir, os.path.relpath(file_path, directory))
                    archive.write(file_path, arcname)
                    pbar.update(1)
    print(f"Directory zipped as: {zip_file}")


if __name__ == "__main__":
    current_directory = os.getcwd()
    # delete_log_files(current_directory)
    zip_directory(current_directory)
