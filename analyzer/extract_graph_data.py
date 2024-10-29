import cv2
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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

def predict_next_points(data_points, num_predictions=5, degree=3):
    x = np.arange(len(data_points)).reshape(-1, 1)
    y = data_points[:, 1].reshape(-1, 1)

    poly_features = PolynomialFeatures(degree=degree)
    x_poly = poly_features.fit_transform(x)

    model = LinearRegression()
    model.fit(x_poly, y)

    predicted_points = []
    for i in range(1, num_predictions + 1):
        next_x = np.array([[len(data_points) + i]])
        next_x_poly = poly_features.transform(next_x)
        next_y = model.predict(next_x_poly)[0, 0]
        predicted_points.append((len(data_points) + i, next_y))

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
    first_derivative = np.gradient(y)
    second_derivative = np.gradient(first_derivative)
    
    return first_derivative, second_derivative

def print_derivatives_and_inflection_points(data_points):
    first_derivative, second_derivative = calculate_derivatives(data_points)
    inflection_points = find_inflection_points(data_points)
    
    print("First Derivative (Rate of Change) for all points:")
    for i, value in enumerate(first_derivative):
        print(f"Point {i}: {value}")
    
    print("\nSecond Derivative (Curvature) for all points:")
    for i, value in enumerate(second_derivative):
        print(f"Point {i}: {value}")
    
    print("\nInflection Points (Index):")
    print(inflection_points)
