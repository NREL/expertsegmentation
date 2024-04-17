from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
    author="Nina Prakash",
    author_email='Nina.Prakash@nrel.gov',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Manufacturing',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Image Processing',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Environment :: GPU',
        'Environment :: GPU :: NVIDIA CUDA',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    description="ExpertSegmentation is a tool for segmenting microscopy with expert-user informed domain targets",
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='expertsegmentation',
    name='expertsegmentation',
    packages=find_packages(include=['expertsegmentation', 'expertsegmentation.*']),
    url='https://github.com/NREL/expertsegmentation',
    version='1.0',
    zip_safe=False,
)