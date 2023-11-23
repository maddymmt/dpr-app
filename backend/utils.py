def get_user_dir(userid):
    # Function to format the USER_DIR path based on the user ID
    return Path(f"users/{userid}")

def clear_directory(path):
    for file in Path(path).glob('*'):
        file.unlink()