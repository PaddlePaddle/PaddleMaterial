import os

import numpy as np
import setuptools
from Cython.Build import cythonize
from setuptools import Extension
from setuptools.command.build_py import build_py

"""
Setup configuration
"""


extensions = [
    Extension(
        "ppmat.models.mattersim.threebody_indices",
        ["ppmat/models/mattersim/threebody_indices.pyx"],
        include_dirs=[np.get_include()],
    )
]


class BuildPyWithGMTNetConfig(build_py):
    """Include the formal GMTNet config as an installed package resource."""

    def run(self):
        super().run()
        destination = os.path.join(
            self.build_lib, "ppmat", "models", "gmtnet", "config.yaml"
        )
        self.mkpath(os.path.dirname(destination))
        self.copy_file(
            "property_prediction/configs/gmtnet/config.yaml",
            destination,
        )


def get_readme() -> str:
    """get README"""
    with open("README.md", encoding="utf-8") as f:
        return f.read()


def get_requirements() -> list:
    """get requirements from PaddleMaterials/requirements.txt"""
    req_list = []
    with open("requirements.txt", "r") as f:
        req_list = f.read().splitlines()
    return req_list


if __name__ == "__main__":
    setuptools.setup(
        name="ppmat",
        author="PaddlePaddle",
        url="https://github.com/PaddlePaddle/PaddleMaterials",
        description=(
            "PaddleMaterials is a data-driven deep learning toolkit based on "
            "PaddlePaddle for material science."
        ),
        long_description=get_readme(),
        long_description_content_type="text/markdown",
        packages=setuptools.find_namespace_packages(
            include=("ppmat", "ppmat.*"),
            exclude=(
                "docs",
                "examples",
                "jointContribution",
                "test",
                "interatomic_potentials",
                "property_prediction",
                "structure_generation",
            ),
        )
        + ["property_prediction"],
        package_data={
            "ppmat.datasets": [
                "gmtnet_dielectric_split_seed32.json",
            ],
        },
        exclude_package_data={
            "": ["*.pdparams", "*.pkl"],
        },
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        install_requires=get_requirements(),
        use_scm_version=True,
        setup_requires=["setuptools_scm"],
        cmdclass={"build_py": BuildPyWithGMTNetConfig},
        ext_modules=cythonize(extensions),
    )
