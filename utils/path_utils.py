import os


def combine_path(path: str, *paths: str, create_if_missing: bool = False) -> str:
    new_path = os.path.join(path, *paths)
    if not os.path.exists(new_path):
        if create_if_missing:
            os.mkdir(new_path)
        else:
            raise Exception(f'Path "{new_path}" does not exist')

    return new_path
