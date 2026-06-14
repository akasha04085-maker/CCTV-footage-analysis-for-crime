import cv2


def load_image(image_path):

    # Read image
    image = cv2.imread(image_path)

    # Check whether image was loaded
    if image is None:
        print("Error: Unable to load image.")
        return None

    # Get dimensions
    height, width, channels = image.shape

    # Display information
    print("Height :", height)
    print("Width :", width)
    print("Channels :", channels)

    print("Type :", type(image))

    print("Datatype :", image.dtype)

    # Print one pixel value
    print("Pixel [100,100] :", image[100, 100])

    return image