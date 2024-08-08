import setuptools

with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()
setuptools.setup(
    name="gemnet_pytorch",
    version="1.0",
    description="GemNet: Universal Directional Graph Neural Networks for Molecules",
    author="Johannes Gasteiger, Florian Becker, Stephan Günnemann",
    author_email="j.gasteiger@in.tum.de",
    packages=["gemnet"],
    install_requires=install_requires,
    zip_safe=False,
    python_requires=">=3.8",
)
