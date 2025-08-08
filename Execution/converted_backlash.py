from pathlib import Path

def converted_backlash(path_str):
    return str(Path(path_str).resolve())
