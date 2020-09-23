import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="schrodinger",
    version="0.0.1",
    author="Jan-Luca Hubald, Nils-Erik Schütte",
    author_email="janluca@uni-bremen.de, nilserik@uni-bremen.de",
    description="This package solves the one dimensional timeindependent schrodinger equation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nilseriks/schrodinger",
    packages=setuptools.find_packages(exlude=('tests*')),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    		],
    scripts=[
	     'solver',
	     'visualizer',
	    ],
    python_requires='>=3.6'
)