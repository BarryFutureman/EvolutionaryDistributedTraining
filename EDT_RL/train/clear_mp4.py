import os
import glob

def clear_mp4_files():
    mp4_files = glob.glob("*.mp4")
    for file in mp4_files:
        os.remove(file)
        print(f"Deleted: {file}")

if __name__ == "__main__":
    clear_mp4_files()
