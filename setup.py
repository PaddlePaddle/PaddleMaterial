import numpy as np
import setuptools
from Cython.Build import cythonize
from setuptools import Extension

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


def get_readme() -> str:
    """get README"""
    with open("README.md", encoding="utf-8") as f:
        return f.read()


def get_requirements() -> list:
    """get requirements from PaddleMaterial/requirements.txt"""
    req_list = []
    with open("requirements.txt", "r") as f:
        req_list = f.read().splitlines()
    return req_list


if __name__ == "__main__":
    setuptools.setup(
        name="ppmat",
        version="0.1.0",  # TODO: remove it when release
        author="PaddlePaddle",
        url="https://github.com/PaddlePaddle/PaddleMaterial",
        description=(
            "PaddleMaterial is a data-driven deep learning toolkit based on "
            "PaddlePaddle for material science."
        ),
        long_description=get_readme(),
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(
            exclude=(
                "docs",
                "examples",
                "jointContribution",
                "test",
                "interatomic_potentials",
                "property_prediction",
                "structure_generation",
            )
        ),
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
        ext_modules=cythonize(extensions),
    )
