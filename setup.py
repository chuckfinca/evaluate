from setuptools import setup, find_packages

setup(
    name="evaluate",
    version="0.1",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        # List your project dependencies here
    ],
)