import subprocess

def get_git_commit() -> str:
    """
    Return the current Git commit hash, or 'unknown' if not available.
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"

def is_git_dirty() -> bool:
    try:
        output = subprocess.check_output(["git", "status", "--porcelain"])
        return len(output.strip()) > 0
    except Exception:
        return False
