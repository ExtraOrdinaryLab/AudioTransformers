from setuptools import find_packages, setup


setup(
    name='audio_transformers', 
    version='0.0.1', 
    description='Open-source library of AudioTransformers', 
    author='Yang Wang', 
    author_email='yangwang4work@gmail.com', 
    package_dir={'': 'src'}, 
    packages=find_packages('src'), 
    python_requires='>=3.9.0', 
    keywords='audio transformers contrastive'
)