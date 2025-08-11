import subprocess
import shutil
from pathlib import Path
import inspect
from types import ModuleType
from typing import Any, Optional, Union

def _resolve_start_path(repo_hint: Optional[Union[str, Path]], obj: Optional[Any]) -> Path:
    """
    Resolve a filesystem path to start Git lookups from.
    Priority:
      1) repo_hint if provided
      2) the file where `obj` is defined (module/class/function)
      3) this file's location (utils module)
    """
    if repo_hint is not None:
        return Path(repo_hint).expanduser().resolve()
    if obj is not None:
        # If it's a module, prefer its __file__
        if isinstance(obj, ModuleType) and getattr(obj, "__file__", None):
            return Path(obj.__file__).resolve()
        # Otherwise, try to locate the defining file (classes, functions, etc.)
        try:
            return Path(inspect.getfile(obj)).resolve()
        except TypeError:
            pass
    # Fallback: the location of this utility file
    return Path(__file__).resolve()


def get_git_commit(
    repo_hint: Optional[Union[str, Path]] = None,
    obj: Optional[Any] = None,
) -> str:
    """
    Return the current Git commit hash for the repository that contains `repo_hint` or `obj`.
    Falls back to the location of this file if neither is provided.
    Returns "unknown" if Git is unavailable or the path is not inside a Git repository.

    Parameters
    ----------
    repo_hint : Optional[Union[str, Path]]
        A path inside the target repository (e.g., Path(__file__).parent).
    obj : Optional[Any]
        A module/class/function whose defining file is inside the target repository.

    Examples
    --------
    # Inside the module where GNS is defined:
    git_commit = get_git_commit(repo_hint=Path(__file__).parent)

    # Or from the class object itself:
    git_commit = get_git_commit(obj=GNS)
    """
    # Quick exit if git is not installed
    if shutil.which("git") is None:
        return "unknown"

    start_path = _resolve_start_path(repo_hint, obj)
    workdir = start_path if start_path.is_dir() else start_path.parent

    # Check if `workdir` is inside a Git work tree
    try:
        probe = subprocess.run(
            ["git", "-C", str(workdir), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, check=True
        )
        if probe.stdout.strip().lower() != "true":
            return "unknown"
    except subprocess.CalledProcessError:
        return "unknown"

    # Get the commit hash (silence all output)
    try:
        res = subprocess.run(
            ["git", "-C", str(workdir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True
        )
        commit = res.stdout.strip()
        return commit if commit else "unknown"
    except subprocess.CalledProcessError:
        return "unknown"


def is_git_dirty() -> bool:
    try:
        output = subprocess.check_output(["git", "status", "--porcelain"])
        return len(output.strip()) > 0
    except Exception:
        return False
