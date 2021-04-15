from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    author="rdnfn",
    author_email="author@example.com",
    description="This project aims to provide environments to train RL agents to control batteries in solar photovoltaic installations.",
    url="url-to-github-page",
    packages=find_packages(),
    test_suite="src.tests.test_all.suite",
)
