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
    install_requires=[
        'iisignature==0.24',
        'pandas==2.0.2',
        'numpy==1.25.2',
        'matplotlib==3.7.1',
        'scipy==1.10.1',
        'scikit-learn==1.2.2',
        'DateTime==5.1',
        'h5py==3.8.0',
        'xgboost==1.7.5',
        'seaborn==0.12.2'
    ]
)
