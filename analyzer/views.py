from django.shortcuts import render
from .forms import GraphUploadForm
from .extract_graph_data import (
    extract_graph_data, analyze_data, predict_next_points, 
    trend_analysis, find_inflection_points, calculate_derivatives
)
import os
import numpy as np
from graph_analyzer.settings import BASE_DIR

def upload_image(request):
    MAX_DISPLAY = 10  

    if request.method == "POST":
        form = GraphUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES["image"]
            path = os.path.join(BASE_DIR, 'images')
            if not os.path.exists(path):
                os.makedirs(path)
            image_path = os.path.join(path, file.name)
            with open(image_path, "wb") as f:
                f.write(file.read())

            data_points = extract_graph_data(image_path)
            minima, maxima = analyze_data(data_points)
            predicted_points = predict_next_points(data_points)

            slope, intercept = trend_analysis(data_points)
            inflection_points = find_inflection_points(data_points)
            derivatives = calculate_derivatives(data_points)

            context = {
                'minima': minima.tolist() if isinstance(minima, np.ndarray) else minima,
                'maxima': maxima.tolist() if isinstance(maxima, np.ndarray) else maxima,
                'predicted_points': [tuple(point) for point in predicted_points],
                'trend': {'slope': slope, 'intercept': intercept},
                'inflection_points': inflection_points.tolist()[:MAX_DISPLAY] if isinstance(inflection_points, np.ndarray) else inflection_points[:MAX_DISPLAY],
                'derivatives': derivatives.tolist()[:MAX_DISPLAY] if isinstance(derivatives, np.ndarray) else derivatives[:MAX_DISPLAY],
                'total_inflection_points': len(inflection_points),
                'total_derivatives': len(derivatives)
            }

            return render(request, 'analyzer/index.html', context)

    else:
        form = GraphUploadForm()

    return render(request, 'analyzer/index.html', {'form': form})
