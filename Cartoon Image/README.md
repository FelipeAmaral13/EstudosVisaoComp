# Cartoonizer

A simple class to cartoonize an input image. The class uses OpenCV, NumPy, Matplotlib, and pathlib Python libraries to process the image.

# Dependencies

    Python 3.x
    OpenCV
    NumPy
    Matplotlib

# Installation

Use pip to install the required libraries:

pip install opencv-python numpy matplotlib

# Usage

To use the Cartoonizer class, create an instance of the class with the input image path as an argument. Then, apply the image processing methods to generate a cartoonized image. Finally, call the show_steps() method to display the steps of the cartoonization process and the final result.

Example:

from Cartoonizer import Cartoonizer

if __name__ == "__main__":
    cartoonizer = Cartoonizer('input_image.png')

    cartoonizer.resize(800, 800)
    cartoonizer.convert_to_gray()
    cartoonizer.apply_gaussian_blur((3, 3), 0)
    cartoonizer.detect_edges(5, 150)
    cartoonizer.edge_preserving_filter(50, 0.4)
    cartoonizer.stylize_image(150, 0.25)

    cartoonizer.show_steps()

# Class methods

    * __init__(self, input_path: str): Initializes the class with the input image path.
    * resize(self, width: int, height: int): Resizes the image to the specified width and height.
    * convert_to_gray(self): Converts the image to grayscale.
    * apply_gaussian_blur(self, kernel_size: tuple, sigma: float): Applies a Gaussian blur filter to the image with the specified kernel size and sigma value.
    * detect_edges(self, ksize: int, threshold: int): Detects the edges of the image using the Laplacian operator and a threshold value.
    * edge_preserving_filter(self, sigma_s: int, sigma_r: float): Applies an edge-preserving filter to the image with the specified sigma_s and sigma_r values.
    * stylize_image(self, sigma_s: int, sigma_r: float): Stylizes the image with the specified sigma_s and sigma_r values.
    * show_image(self, img, title): Displays the specified image with the specified title.
    * show_steps(self): Displays the steps of the cartoonization process and the final result.

# License

This project is licensed under the MIT License.
![Capturar2](https://user-images.githubusercontent.com/5797933/130156526-19b87594-62a7-4220-a739-d9c19951157f.PNG)
