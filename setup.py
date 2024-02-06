from setuptools import setup, find_packages

setup(
    name='sentiment-analysis-for-msci-rating',
    version='0.1.0',
    author='Muhammad Azeem ARSHAD',
    author_email='Muhammad.arshad.2@etu.unige.ch',
    packages=find_packages(where='esg_classification'),
    package_dir={'': 'esg_classification'},
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'torch',
        'transformers',
        'spacy',
        'datasets',
        'tqdm'        
    ],
    python_requires='>=3.10',
    description='Système de notation automatique de durabilité de quelques communes suisses',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/azeemarshd/sentiment-analysis-for-msci-rating',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
        ],
)
