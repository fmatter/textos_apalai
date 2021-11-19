from setuptools import setup


setup(
    name='textosapalai',
    py_modules=['textosapalai'],
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'cldfbench.dataset': [
            'textos_apalai=cldfbench_textosapalai:Dataset',
        ]
    },
    install_requires=[
        'cldfbench',
    ],
    extras_require={
        'test': [
            'pytest-cldf',
        ],
    },
)
