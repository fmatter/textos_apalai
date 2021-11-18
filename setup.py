from setuptools import setup


setup(
    name='cldfbench_cldftest',
    py_modules=['cldfbench_cldftest'],
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'cldfbench.dataset': [
            'cldftest=cldfbench_cldftest:Dataset',
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
