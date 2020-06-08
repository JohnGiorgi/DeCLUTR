import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="declutr",
    version="0.1.0",
    author="John Giorgi",
    author_email="johnmgiorgi@gmail.com",
    description=("DeCLUTR: Deep Contrastive Learning for Unsupervised Textual Representations."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JohnGiorgi/DeCLUTR",
    packages=setuptools.find_packages(),
    keywords=[
        "universal sentence embeddings" "contrastive learning",
        "representation learning",
        "textual representations",
        "natural language processing",
        "allennlp",
        "transformers",
        "pytorch",
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
    install_requires=["torch>=1.5.0", "pytorch-metric-learning>=0.9.85", "typer>=0.2.1"],
    extras_require={"dev": ["black", "flake8", "hypothesis", "pytest"]},
)
