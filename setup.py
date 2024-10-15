from setuptools import setup, find_packages

setup(
    name="finca",
    version="0.1",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        'python-dotenv>=1.0.1',
        'platformdirs>=4.2.2',
        'logger>=1.4',
        'dspy-ai>=2.5.7'
    ],
)