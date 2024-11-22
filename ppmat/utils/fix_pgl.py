import os

import pkg_resources


def get_library_path(library_name):
    distribution = pkg_resources.get_distribution(library_name)
    location = distribution.location
    return os.path.abspath(location)


def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content


def write_file(file_path, content):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


library_path = get_library_path("pgl")
file_paths = ["pgl/math.py", "pgl/utils/helper.py", "pgl/utils/op.py"]

# replace "fluid" with "base"
for file_path in file_paths:
    full_path = os.path.join(library_path, file_path)
    print(f"Processing {full_path}, replacing 'fluid' with 'base'...")
    content = read_file(full_path)
    new_content = content.replace("paddle.fluid", "paddle.base")
    new_content = new_content.replace("paddle.base.core as core", "paddle.base as core")
    new_content = content.replace(
        "from paddle.base.layers import core", "from paddle.base import core"
    )
    write_file(full_path, new_content)

# delete "overwrite" paramters in "pgl/utils/helper.py"
file_paths = ["pgl/utils/helper.py"]
for file_path in file_paths:
    full_path = os.path.join(library_path, file_path)
    print(f"Processing {full_path}, deleting 'overwrite' paramters...")
    content = read_file(full_path)
    new_content = content.replace(
        "return _C_ops.scatter(x, index, updates, 'overwrite', overwrite)",
        "return _C_ops.scatter(x, index, updates, overwrite)",
    )
    write_file(full_path, new_content)
