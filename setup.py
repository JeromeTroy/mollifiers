from setuptools import setup, find_packages
import pathlib 

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="mollifiers",
    version="1.0.0",
    description="Simple extensible mathematical mollifier and friends",
    long_description=long_description, 
    long_description_content_type="text/markdown",
    url="TODO",
    author="Jerome Troy",
    author_email="jrtroy@udel.edu",
    packages=find_packages(where="src"),
    python_requires=">=3.7, <4",
    install_requires=["numpy", "scipy"]
)