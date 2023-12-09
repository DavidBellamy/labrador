from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

setup(
    name="lab_transformers",
    version="0.1",
    packages=find_packages(),
    install_requires=required_packages,
    python_requires='>=3.10',
    author="David R. Bellamy",
    author_email="bellamyrd@gmail.com",
    description="A package for experimenting with Labrador and BERT models, which were pre-trained on the lab data in MIMIC-IV.",
    url="https://github.com/yourusername/lab_transformers",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
