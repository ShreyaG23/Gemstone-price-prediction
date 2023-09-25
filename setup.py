from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e ."

def GetRequirement(path: str)->List[str]:
    requirements = []
    with open(path) as file:
        requirements = file.readlines()
        requirements = [requires.replace("\n", "") for requires in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements
    

setup(
    name = "Gmstone predidction",
    version = "0.0.1",
    author = "shreya gupta",
    author_email = "shreygold27@gmail.com",
    packages=find_packages(),
    install_requires=GetRequirement("requirements.txt")
)