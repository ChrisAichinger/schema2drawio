from setuptools import setup

setup(
    name='schema2drawio',
    version='0.1.0',
    description="Generate SQL schema diagrams in Draw.IO format",
    long_description=open('README.rst').read(),
    author="Christian Aichinger",
    author_email="Greek0@gmx.net",
    url='https://github.com/Grk0/schema2drawio',
    download_url='https://github.com/Grk0/schema2drawio/tarball/0.1.0',
    license="MIT",
    py_modules=['schema2drawio'],
    entry_points={
        'console_scripts': [
            'schema2drawio = schema2drawio:main',
        ],
    },
    keywords=['database', 'sql', 'schema', 'entity-relation', 'uml', 'draw.io'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: System Administrators',
        'Environment :: Console',
        'Topic :: Database',
        'Topic :: Documentation',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Software Development :: Documentation',
        'Topic :: Utilities',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
