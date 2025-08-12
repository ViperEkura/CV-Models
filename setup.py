from setuptools import setup, find_packages
import re

with open("requirements.txt") as f:
    required = [line for line in f.read().splitlines() 
                if line and re.match(r'^[^=]+==[^=]+$', line.strip())]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cv_models",
    version="1.0.0",
    author="ViperEkura",
    author_email="viper.ekura@example.com",
    description="A collection of computer vision models including DETR, MOTR and related components",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ViperEkura/cv_models",
    packages=find_packages(),
    install_requires=required,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            "cv_models=modules.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)