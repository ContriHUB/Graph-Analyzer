from django.shortcuts import render
from .forms import GraphUploadForm
from .extract_graph_data import extract_graph_data, analyze_data, predict_next_points
import os
from graph_analyzer.settings import BASE_DIR

def upload_image(request):
    if request.method == "POST":
        form = GraphUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES["image"]
            path = os.path.join(BASE_DIR,'images')
            if not os.path.exists(path):
                os.makedirs(path)
            image_path = os.path.join(path, file.name)
            with open(image_path, "wb") as f:
                f.write(file.read())

            # Process image and get graph data
            data_points = extract_graph_data(image_path)
            minima, maxima = analyze_data(data_points)
            predicted_points = predict_next_points(data_points)

            context = {
                'minima': minima,
                'maxima': maxima,
                'predicted_points': predicted_points
            }
            return render(request, 'analyzer/index.html', context)
    else:
        form = GraphUploadForm()
    return render(request, 'analyzer/index.html', {'form': form})