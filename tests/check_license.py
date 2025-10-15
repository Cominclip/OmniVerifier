

import sys
from pathlib import Path




def main():
    path_list: list[Path] = []
    for check_dir in sys.argv[1:]:
        path_list.extend(Path(check_dir).glob("**/*.py"))

    for path in path_list:
        with open(path.absolute(), encoding="utf-8") as f:
            file_content = f.read().strip().split("\n")
            license = "\n".join(file_content[:5])
            if not license:
                continue

            print(f"Check license: {path}")
            assert all(keyword in license for keyword in KEYWORDS), f"File {path} does not contain license."


if __name__ == "__main__":
    main()
