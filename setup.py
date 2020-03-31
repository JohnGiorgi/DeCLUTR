import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="t2t",
    version="0.1.0",
    author="John Giorgi",
    author_email="johnmgiorgi@gmail.com",
    description=(
        "A contrastive, self-supervised method for learning textual representations."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JohnGiorgi/t2t",
    packages=setuptools.find_packages(),
    keywords=[
        "natural language processing",
        "pytorch",
        "allennlp",
        "transformers",
        "contrastive learning",
        "textual representations",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.4.0",
        "pytorch-metric-learning>=0.9.79",
        "typer>=0.1.1",
    ],
    extras_require={"dev": ["black", "flake8", "pytest"]},
)
