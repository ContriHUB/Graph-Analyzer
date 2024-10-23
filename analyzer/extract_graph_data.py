import cv2
import numpy as np
from scipy import stats
from scipy.signal import find_peaks

def extract_graph_data(image_path):
    img = cv2.imread(image_path, 0)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    graph_points = []
    for contour in contours:
        for point in contour:
            graph_points.append(point[0])
    return np.array(graph_points)

def analyze_data(data_points):
    maxima = np.argmax(data_points, axis=0)
    minima = np.argmin(data_points, axis=0)
    return minima, maxima

def predict_next_points(data_points, num_predictions=5):
    x = np.arange(len(data_points))
    y = data_points[:, 1]
    slope, intercept, _, _, _ = stats.linregress(x, y)
    last_x = len(data_points)
    predicted_points = []
    for i in range(num_predictions):
        next_x = last_x + i + 1
        next_y = slope * next_x + intercept
        predicted_points.append((next_x, next_y))
    return predicted_points

def trend_analysis(data_points):
    x = np.arange(len(data_points))
    y = data_points[:, 1]
    trend = np.polyfit(x, y, 1)
    slope, intercept = trend
    return slope, intercept

def find_inflection_points(data_points):
    y = data_points[:, 1]
    second_derivative = np.diff(np.sign(np.diff(y)))
    inflection_points = np.where(second_derivative != 0)[0] + 1
    return inflection_points

def calculate_derivatives(data_points):
    y = data_points[:, 1]
    derivatives = np.diff(y)
    return derivatives
