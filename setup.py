import setuptools
from typing import List
import glob
from Cython.Build import cythonize
import numpy as np

import my_ml


def get_scripts_from_bin() -> List[str]:
    """Get all local scripts from bin so they are included in the package."""
    return glob.glob("bin/*")


def get_package_description() -> str:
    """Returns a description of this package from the markdown files."""
    with open("README.md", "r") as stream:
        readme: str = stream.read()
    return readme


setuptools.setup(
    name="my_ml",
    version=my_ml.__version__,
    author="Colin Manko",
    author_email="colin@colinmanko.com",
    description="ML Algos from Scratch, implemented in Python and Numpy.",
    long_description=get_package_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/big-c-note/my_ml_from_scratch",
    ext_modules=cythonize("my_ml/model/_split_data_fast.pyx"),
    include_dirs=[np.get_include()],
    zip_safe=False,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    scripts=get_scripts_from_bin(),
    python_requires=">=3.7",
)
