# Graph Analyzer

Graph Analyzer is a Django-based web application that allows users to upload an image of a graph, extract the underlying data, and analyze it for key features such as minima, maxima, and predicted future values. This tool is useful for visualizing network traffic or analyzing real-world data represented in graph form, including the scientific and financial data.

## Installation and Setup

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- pip (Python package manager)
- Django 4.x
- OpenCV 4.x
- NumPy
- scikit-learn

### Clone the Repository

```bash
git clone https://github.com/<your_username>/Graph-Analyzer.git
cd Graph-Analyzer
```

### Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Apply Migrations

```bash
python manage.py migrate
```

### Run the Development Server

```bash
python manage.py runserver
```
Visit `http://127.0.0.1:8000/` in your browser to start uploading graph images.

## Usage

1. **Upload a Graph Image**: Navigate to the home page and choose an image file that contains a graph.
2. **Graph Analysis**: The application will process the image, extract data points, and find the minima, maxima, and predict the next 5 values.
3. **View Results**: Once processed, the results will display the detected minima, maxima, and predicted values.

## Example Workflow

1. **Step 1**: Upload an image of a line graph showing time vs network traffic.
2. **Step 2**: The application extracts the data points from the image.
3. **Step 3**: Minima, maxima, and the next 5 values are calculated and displayed on the home page.

## Troubleshooting

If you encounter issues with the image not loading properly or the image processing functions failing, ensure that:
- The image is correctly uploaded.
- The image path is accessible to the application (check permissions and file locations).

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m "Add some feature"`)
4. Push to the branch (`git push origin feature-name`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
