from setuptools import setup, find_packages

setup(
    name="apple-ocr",
    version="1.0.0",
    license="Apache License 2.0",
    author="Louis Brulé Naudet",
    author_email="louisbrulenaudet@icloud.com",
    description="An OCR (Optical Character Recognition) utility for text extraction from images.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/louisbrulenaudet/apple-ocr",
    homepage="https://github.com/louisbrulenaudet/apple-ocr",
    project_urls={"repository": "https://github.com/louisbrulenaudet/apple-ocr"},
    keywords="OCR, image-recognition, text-extraction, clustering, Apple Vision, NLP, LLM, data",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "Pillow",
        "scikit-learn",
        "plotly",
        "pyobjc",
    ],
)