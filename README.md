# Easy-to-Use Apple Vision wrapper for text extraction and clustering
[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg)](https://badge.fury.io/py/tensorflow) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![Maintainer](https://img.shields.io/badge/maintainer-@louisbrulenaudet-blue)

`apple_ocr` is a utility for Optical Character Recognition (OCR) that facilitates the extraction of text from images. This Python-based tool is designed to help developers, researchers, and enthusiasts in the field of text extraction and clustering. It leverages a combination of various technologies to achieve this, including the Vision framework provided by Apple.

![Plot](https://github.com/louisbrulenaudet/apple-ocr/blob/main/scatter.png?raw=true)

## Features
- **Text Recognition**: `apple_ocr` uses the Vision framework to recognize text within an image. It extracts recognized text and provides information about its confidence levels.

- **Clustering**: The tool can perform K-Means clustering on the extracted data. It groups similar text elements together based on their coordinates.

- **Interactive 3D Visualization**: `apple_ocr` offers an interactive 3D scatter plot using Plotly, displaying the clustered text elements. This visualization helps users gain insights into the distribution of text and text density.

## Dependencies
The script relies on the following Python libraries:
- Torch
- NumPy
- Pandas
- Pillow
- Scikit-learn
- Plotly
- Pyobjc

## Usage
Here's how you can use `apple_ocr`:

1. **Installation**: Install the required libraries, including `Torch`, `NumPy`, `Pandas`, `Pillow`, `scikit-learn`, and `Plotly`.

2. **Initialization**: Create an instance of the `OCR` class, providing an image to be processed.
```python
from apple_ocr.ocr import OCR
from PIL import Image

image = Image.open("your_image.png")
ocr_instance = OCR(image=image)
```

3. **Text Recognition**: Use the `recognize` method to perform text recognition. It will return a structured DataFrame containing recognized text, bounding box dimensions, text density, and centroid coordinates.
```python
dataframe = ocr_instance.recognize()`
```

4. **Clustering**: Use the `cluster` method to perform K-Means clustering on the recognized text data. This method assigns cluster labels to each data point based on their coordinates.
```python
cluster_labels = ocr_instance.cluster(dataframe, num_clusters=3)
```

5. **Visualization**: Finally, use the `scatter` method to create an interactive 3D scatter plot. This plot visualizes the clustered text elements, including centroids, text density, and more.
```python
ocr_instance.scatter()
```

## Example
Here's an example of the entire process:

```python
from apple_ocr.ocr import OCR
from PIL import Image

image = Image open("your_image.png")
ocr_instance = OCR(image=image)
dataframe = ocr_instance.recognize()
cluster_labels = ocr_instance.cluster(dataframe, num_clusters=3)
ocr_instance.scatter()
```

## Citing this project
If you use this code in your research, please use the following BibTeX entry.

```BibTeX
@misc{louisbrulenaudet2023,
	author = {Louis Brulé Naudet},
	title = {Easy-to-Use Apple Vision wrapper for text extraction and clustering},
	howpublished = {\url{https://github.com/louisbrulenaudet/apple-ocr}},
	year = {2023}
}

```
## Feedback
If you have any feedback, please reach out at [louisbrulenaudet@icloud.com](mailto:louisbrulenaudet@icloud.com).