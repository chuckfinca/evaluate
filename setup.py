from setuptools import setup, find_packages

setup(
    name="evaluate",
    version="0.1",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        'python-dotenv>=1.0.1',
    ],
)