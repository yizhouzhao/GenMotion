import setuptools
from genmotion import __version__

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GenMotion",
    version=__version__,
    author="Yizhou Zhao, Wensi Ai",
    author_email="yizhouzhao@ucla.edu, va0817@ucla.edu",
    description="Deep motion generator collections",
    long_description=long_description,
    url="https://github.com/yizhouzhao/GenMotion",
    project_urls={
        "Bug Tracker": "https://github.com/yizhouzhao/GenMotion/issues",
        "Documentation": "https://genmotion.readthedocs.io/en/latest"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=[package for package in setuptools.find_packages() if package.startswith('genmotion')],
    python_requires=">=3.6",
)
