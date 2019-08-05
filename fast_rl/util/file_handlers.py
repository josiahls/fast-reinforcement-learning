import os
import queue
from pathlib import Path, PosixPath


def get_absolute_path(directory, project_root='fast_rl', ignore_hidden=True, ignore_files=True):
    """
    Gets the absolute path to a directory in the project structure using depth first search.

    Args:
        directory: The name of the folder to look for.
        project_root: The project root name. Generally should not be changed.
        ignore_hidden: For future use, for now, throws error because it cannot handle hidden files.
        ignore_files: For future use, for now, throws error because it is expecting directories.
    Returns:
    """
    if not ignore_hidden:
        raise NotImplementedError('ignore_hidden is not supported currently.')
    if not ignore_files:
        raise NotImplementedError('ignore_files is not supported currently.')

    full_path = Path(__file__).parents[0]  # type: PosixPath

    # Move up the path address to the project root
    while full_path.name != project_root:
        full_path = full_path.parents[0]

    if full_path.name == directory: return str(full_path)
    # Find the path to the directory
    searched_directory = queue.Queue()
    searched_directory.put_nowait(full_path)
    # Will do a breadth first search
    while not searched_directory.empty():
        full_path_str = str(searched_directory.get_nowait())
        if os.path.exists(os.path.join(full_path_str, directory)):
            return os.path.join(full_path_str, directory)

        for inner_dir in os.listdir(full_path_str):
            if str(inner_dir).__contains__('.'):
                continue  # Directory is either a file, or hidden. Skip this

            searched_directory.put_nowait(os.path.join(full_path_str, inner_dir))

    raise IOError(f'Path to {directory} not found.')