import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wikiled.nlp",
    version="0.0.1",
    author="Wikiled",    
    description="Nlp Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AndMu/Wikiled.Python.Nlp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)