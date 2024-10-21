import cv2
import numpy as np
from scipy import stats

def extract_graph_data(image_path):
    # Read the image
    img = cv2.imread(image_path, 0)

    # Pre-process the image (blurring, edge detection, etc.)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours of the graph
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract coordinates (or more sophisticated image analysis)
    graph_points = []
    for contour in contours:
        for point in contour:
            graph_points.append(point[0])

    # Return data points (this should be converted to meaningful graph data)
    return np.array(graph_points)

def analyze_data(data_points):
    # Find minima and maxima
    maxima = np.argmax(data_points, axis=0)
    minima = np.argmin(data_points, axis=0)

    return minima, maxima

def predict_next_points(data_points, num_predictions=5):
    # Convert data points to a time series (assuming equal time intervals)
    x = np.arange(len(data_points))
    y = data_points[:, 1]  # Assuming y-coordinates represent the data values

    # Perform linear regression
    slope, intercept, _, _, _ = stats.linregress(x, y)

    # Predict next points
    last_x = len(data_points)
    predicted_points = []
    for i in range(num_predictions):
        next_x = last_x + i + 1
        next_y = slope * next_x + intercept
        predicted_points.append((next_x, next_y))

    return predicted_points