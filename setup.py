from setuptools import setup, find_packages

setup(
    name="memoryweave",
    version="0.1.0",
    description="Persistent, self-organizing memory system for AI agents",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="MukundaKatta",
    url="https://github.com/MukundaKatta/memoryweave",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["numpy>=1.24"],
    extras_require={
        "full": ["chromadb>=0.4", "sentence-transformers>=2.2", "networkx>=3.0", "faiss-cpu>=1.7"],
        "dev": ["pytest>=7.0"],
    },
)
