import os


def create_directories():
    directories = {
        "data": ["prediction", "preprocessed", "stock"],
        "models": [],
    }
    
    for main_dir, sub_dirs in directories.items():
        os.makedirs(main_dir, exist_ok=True)
        for sub_dir in sub_dirs:
            os.makedirs(os.path.join(main_dir, sub_dir), exist_ok=True)
        print(f"Created directory: {main_dir} with subdirectories: {', '.join(sub_dirs) if sub_dirs else 'None'}")


def main():
    create_directories()

    print("Directories installation completed successfully!")

if __name__ == "__main__":
    main()
