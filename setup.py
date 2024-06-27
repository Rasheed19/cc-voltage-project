from setuptools import setup, find_packages

setup(
    name="cc-voltage-project",
    version="0.0.1",
    author="Rasheed Ibraheem",
    author_email="R.O.Ibraheem@sms.ed.ac.uk",
    maintainer="Rasheed Ibraheem",
    maintainer_email="R.O.Ibraheem@sms.ed.ac.uk",
    description="""Capacity and Internal Resistance of lithium-ion batteries:
            Full degradation curve prediction from Voltage response at
            constant Current at discharge""",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Rasheed19/cc-voltage-project.git",
    project_urls={
        "Bug Tracker": "https://github.com/Rasheed19/cc-voltage-project.git/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
)
