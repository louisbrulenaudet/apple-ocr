import io
import warnings

import torch
import numpy as np
import pandas as pd

from PIL import Image, ImageDraw
from sklearn.cluster import KMeans

import Vision

import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

class OCR:
    def __init__(self, image:any, format:any="PNG"):
        """
        OCR class to extract text from images.

        Parameters
        ----------
        image : object
            The input image to be converted.
        """
        self.image = image
        self.res = None
        self.format= format
        self.data = []
        self.dataframe = None


    def imageToBuffer(self, image:any) -> any:
        """
        Convert a PIL image to bytes.

        Parameters
        ----------
        image : object
            The input image to be converted.

        Returns
        -------
        bytes
            The image data in bytes.
        """
        buffer = io.BytesIO()
        image.save(buffer, format=self.format)

        return buffer.getvalue()


    def completionHandler(self, request:any, error:any) -> None:
        """
        Handle completion of text recognition request.

        This method processes the results of a text recognition request 
        and extracts recognized text and its confidence levels.

        Parameters
        ----------
        request : object
            The text recognition request object.

        error : object
            Error object, if any.

        Returns
        -------
        None
        """
        observations = request.results()
        results = []

        try:
            for observation in observations:
                recognized_text = observation.topCandidates_(1)[0]
                results.append([recognized_text.string(), recognized_text.confidence()])

        except:
            pass

        return None


    def dealloc(self, request:any, request_handler:any) -> None:
        """
        Clean up and deallocate resources.

        This method is responsible for releasing allocated resources and performing essential 
        cleanup operations related to text recognition tasks.

        Parameters
        ----------
        request : object
            The text recognition request object.

        request_handler : object
            The image request handler object.

        Returns
        -------
        None
        """
        request.dealloc()
        request_handler.dealloc()

        return None


    def cluster(self, dataframe:any, num_clusters:int=3) -> list:
        """
        Perform K-Means clustering on a given DataFrame and assign cluster labels to the data.

        Parameters
        ----------
        dataframe : any
            The input DataFrame containing the data to be clustered.

        num_clusters : int, optional
            The number of clusters to create. Default is 3.

        Returns
        -------
        labels : list
            A list of cluster assignments for each data point in the DataFrame.

        Examples
        --------
        >>> df = pd.DataFrame({
        >>>    "x": [1.2, 2.0, 0.8, 4.5, 3.2],
        >>>    "y": [2.4, 1.8, 3.6, 4.0, 5.0]
        >>> })
        >>> clustering = ClusterClass()
        >>> labels = clustering.cluster(df, num_clusters=2)
        >>> print(labels)
        [1, 1, 0, 1, 0]
        """
        array = np.array([(x, y) for x, y in zip(dataframe["x"], dataframe["y"])])

        # Convert data to PyTorch tensor
        tensor = torch.tensor(array, dtype=torch.float32)

        # Perform K-Means clustering using scikit-learn
        kmeans = KMeans(n_clusters=num_clusters, n_init=10)
        kmeans.fit(tensor)

        # Get the cluster assignments
        labels = kmeans.labels_
        self.dataframe["Cluster"] = labels

        return labels

    
    def scatter(self) -> any:
        """
        Create and display a 3D scatter plot with cluster coloring and text density information.

        This function performs K-Means clustering on the data and then generates a 3D scatter plot 
        using Plotly. The data points are colored based on the assigned clusters, and text density information is displayed on hover.

        Returns
        -------
        None
            This function does not return any value; it displays the plot interactively.
        """
        self.cluster(dataframe=self.dataframe)

        fig = go.Figure(data=[go.Scatter3d(
            x=self.dataframe["x"],
            y=self.dataframe["Density"],
            z=self.dataframe["y"],
            mode="markers",
            marker=dict(
                size=8,
                color=self.dataframe["Cluster"],
                colorscale="bluered",
                colorbar=dict(title="Cluster")),
            text=self.dataframe["Density"],
            hoverinfo="x+y+z+text",
            hovertext=self.dataframe["Content"],
        )])

        fig.add_trace(go.Scatter3d(
            x=self.dataframe["Centroid x"],
            y=self.dataframe["Density"],
            z=self.dataframe["Centroid y"],
            mode="markers",
            marker=dict(
                size=4,
                color=self.dataframe["Density"], 
                colorscale="bluered"
            ),
            text=self.dataframe["Density"],
            name="Centroids density",
            hoverinfo="x+y+z+text",
            hovertext=self.dataframe["Content"],
        ))

        fig.update_layout(title="Vector, Centroid, and Text Density in 3D")
        fig.show()

        return None


    def recognize(self) -> any:
        """
        Perform text recognition using Apple's Vision framework.

        Parameters
        ----------
        None

        Returns
        -------
        self.data : list
            List of tuples containing the text, confidence, and bounding box.
        """
        buffer = self.imageToBuffer(self.image)

        # Create a new image-request handler.
        request_handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
            buffer, None
        )

        # Create a new request to recognize text.
        request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(self.completionHandler)
        request_handler.performRequests_error_([request], None)

        try:
            for observation in request.results():
                bbox = observation.boundingBox()
                w, h = bbox.size.width, bbox.size.height
                x, y = bbox.origin.x, bbox.origin.y
                self.data.append((observation.text(), observation.confidence(), [x, y, w, h]))

        except:
            pass

        self.dealloc(request, request_handler)

        # Unpack the results for better readability
        content, confidences, bbox = zip(*self.data)

        # Extract bounding box dimensions
        w = [w for (x, y, w, h) in bbox]
        h = [h for (x, y, w, h) in bbox]
        x = [x for (x, y, w, h) in bbox]
        y = [y for (x, y, w, h) in bbox]

        # Calculate text density using NumPy
        text_areas = np.array(w) * np.array(h)
        total_area = 1 * 1
        densities = text_areas / total_area

        # Calculate centroids
        cx = np.array(x) + np.array(w) / 2
        cy = np.array(y) + np.array(h) / 2

        self.dataframe = {
            "Content": content,
            "Length": w,
            "Density": densities,
            "x": x,
            "y": y,
            "Centroid x": cx,
            "Centroid y": cy
        }

        return self.dataframe
