from setuptools import setup, find_packages

setup(
    name="llm-failover",
    version="0.2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "httpx>=0.27.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0", "pytest-asyncio>=0.23.0", "pytest-cov>=4.1.0"],
    },
    python_requires=">=3.10",
)
