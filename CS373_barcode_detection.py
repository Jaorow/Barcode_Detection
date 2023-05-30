# Built in packages
import math
import sys
from pathlib import Path

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):
    '''creates a greyscale pixel array of size image_width * image_height, with all values initialized to initValue'''
    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array

# You can add your own functions here:

def streachTo0_255(px_array):
    '''streaches the greyscale pixel array to 0-255'''
    min = 255
    max = 0
    for i in range(len(px_array)):
        for j in range(len(px_array[0])):
            if px_array[i][j] < min:
                min = px_array[i][j]
            if px_array[i][j] > max:
                max = px_array[i][j]
    for i in range(len(px_array)):
        for j in range(len(px_array[0])):
            px_array[i][j] = int((px_array[i][j] - min) * 255 / (max - min))
    return px_array


def computeHistogram(pixel_array, image_width, image_height, nr_bins = 8):
    '''computes the histogram of the pixel array'''
    histogram = [0] * nr_bins
    for i in range(len(pixel_array)):
        for j in range(len(pixel_array[0])):
            histogram[int(pixel_array[i][j])] += 1
    return histogram

def computeCumulativeHistogram(pixel_array, image_width, image_height, nr_bins):
    '''computes the cumulative histogram of the pixel array'''

    histogram = computeHistogram(pixel_array, image_width, image_height, nr_bins)

    cumulative_histogram = [0] * nr_bins
    cumulative_histogram[0] = histogram[0]
    for i in range(len(cumulative_histogram)):
        cumulative_histogram[i] = cumulative_histogram[i - 1] + histogram[i]
    return cumulative_histogram

def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    '''computes the thresholded image'''
    return list(map(lambda x: list(map(lambda y: 255 if y >= threshold_value else 0, x)), pixel_array))


def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    '''computes given RGB Arrays to greyscale array'''
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):
            greyscale_pixel_array[i][j] = int(round(0.299 * pixel_array_r[i][j] + 0.587 * pixel_array_g[i][j] + 0.114 * pixel_array_b[i][j],0))

    return greyscale_pixel_array


def computeMinAndMaxValues(pixel_array, image_width, image_height):
    '''computes the minimum and maximum values of the pixel array'''
    flat_array = [pixel_array[i][j] for i in range(image_height) for j in range(image_width)]
    return (min(flat_array), max(flat_array))

def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    '''scales the greyscale pixel array to 0-255 and quantizes it to 8 bits'''
    min, max = computeMinAndMaxValues(pixel_array, image_width, image_height)
    if min-max == 0:
        return list(map(lambda x: list(map(lambda y: 0, x)), pixel_array))
    return list(map(lambda x: list(map(lambda y: int(round(((y - min) * 255 / (max - min)),0)), x)), pixel_array))


def computeVerticalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    '''computes the vertical edges of the pixel array using the Sobel operator'''
    vertical_edges = createInitializedGreyscalePixelArray(image_width, image_height)
    sobel_x = [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]

    for i in range(image_height):
        for j in range(image_width):

            if i == 0 or i == image_height - 1 or j == 0 or j == image_width - 1:
                # if we are at the border of the image, we set the value to 0
                vertical_edges[i][j] = 0.0

            else:
                # otherwise we apply the Sobel operator
                            pixel_value = 0.0
                            for di in range(-1, 2):
                                for dj in range(-1, 2):
                                    pixel_value += pixel_array[i+di][j+dj] * sobel_x[di+1][dj+1]

                            vertical_edges[i][j] = abs(pixel_value) / 8
    return vertical_edges


def computeHorizontalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    sobel_y = [[-1, -2, -1], 
               [0, 0, 0], 
               [1, 2, 1]]

    output_image = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(1, image_height-1):
        for j in range(1, image_width-1):
            pixel_value = 0.0
            for di in range(-1, 2): 
                for dj in range(-1, 2):
                    pixel_value += pixel_array[i+di][j+dj] * sobel_y[di+1][dj+1]
            output_image[i][j] = abs(pixel_value)/8
    return output_image

def computeBoxAveraging3x3(pixel_array, image_width, image_height):
    output_image = createInitializedGreyscalePixelArray(image_width, image_height)


    for i in range(0, image_height):
        for j in range(0, image_width):
            if i == 0 or i == image_height - 1 or j == 0 or j == image_width - 1:
                # if we are at the border of the image, we set the value to 0
                output_image[i][j] = 0.0
            else:
                output_image[i][j] = abs(pixel_array[i + 1][j - 1] + pixel_array[i + 1][j] + pixel_array[i + 1][j + 1] 
                                        + pixel_array[i][j - 1] + pixel_array[i][j] + pixel_array[i][j + 1] 
                                        + pixel_array[i - 1][j - 1] + pixel_array[i - 1][j] + pixel_array[i - 1][j + 1]) / 9
                                     
    return output_image


def computeGaussianAveraging3x3RepeatBorder(pixel_array, image_width, image_height):
    output_image = createInitializedGreyscalePixelArray(image_width, image_height)
    gaussian_filter = [
                        [1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]
                    ]
    for i in range(image_height):
        for j in range(image_width):
            pixel_value = 0.0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    new_x = max(0, min(di + i, image_height - 1))
                    new_y = max(0, min(dj + j, image_width - 1)) 
                    pixel_value += pixel_array[new_x][new_y] * gaussian_filter[di+1][dj+1]
            output_image[i][j] = round((pixel_value *  1/16),2)
    return output_image

def computeStandardDeviationImage3x3(pixel_array, image_width, image_height):
    output_image = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            mean = 0
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    mean += pixel_array[i + di][j + dj]
            mean /= 9.0
            variance = 0

            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    variance += math.pow(pixel_array[i + di][j + dj] - mean, 2)

            variance /= 9.0

            standard = math.sqrt(variance)
            output_image[i][j] = standard
    
    return output_image

def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    '''computes the thresholded image'''
    return list(map(lambda i: list(map(lambda j: 255 if j >= threshold_value else 0, i)), pixel_array))

def computeDilation8Nbh5x5FlatSE(pixel_array, image_width, image_height):
    output = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if 0 <= i+di < image_height and 0 <= j+dj < image_width:
                        if pixel_array[i + di][j + dj] > 0:
                            output[i][j] = 1
                            break
                if output[i][j] == 1:
                    break
    return output

def computeErosion8Nbh5x5FlatSE(pixel_array, image_width, image_height):
    output = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            misses = False
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if 0 <= i+di < image_height and 0 <= j+dj < image_width:
                        if pixel_array[i+di][j+dj] == 0:
                            # pixel misses
                            misses=True
                            break
                    else:
                        misses=True
                        break
                if misses:
                    break
            if not misses:
                output[i][j] = 1
    return output

# This is our code skeleton that performs the barcode detection.
# Feel free to try it on your own images of barcodes, but keep in mind that with our algorithm developed in this assignment,
# we won't detect arbitrary or difficult to detect barcodes!
def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    filename = "Barcode1"
    input_filename = "images/"+filename+".png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(filename+"_output.png")
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure



    # STUDENT IMPLEMENTATION here

    
    #! STEP 1 Convert to grayscale and normolise
    greyscaled = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    # quontisise and strech
    # TODO: test if this helps
    # greyscaled = scaleTo0And255AndQuantize(greyscaled, image_width, image_height)   


    #! STEP 2A apply sobel filter
    vertical_edges = computeVerticalEdgesSobelAbsolute(greyscaled, image_width, image_height)
    horizontal_edges = computeHorizontalEdgesSobelAbsolute(greyscaled, image_width, image_height)
    # combine the two edges by calculating absolute diffrence
    px_array = [[abs(vertical_edges[i][j] - horizontal_edges[i][j]) for j in range(image_width)] for i in range(image_height)]

    #! STEP 2B apply Standard deviation method
    # TODO: not currently using standard deviation this, will deside later, if need change abs_dif_edges to std
    # blurs the image
    px_array = computeStandardDeviationImage3x3(px_array, image_width, image_height)


    #! STEP 3 apply Gaussian filter
    #!! this takes ages!! 
    for i in range(4):
        px_array = computeGaussianAveraging3x3RepeatBorder(px_array, image_width, image_height)

    contrast_strech = streachTo0_255(px_array)


    #! STEP 4 apply thresholding
    px_array = computeThresholdGE(px_array, 100, image_width, image_height)


    #! STEP 5a apply dilation
    for _ in range(3):
        px_array = computeDilation8Nbh5x5FlatSE(px_array, image_width, image_height)

    #! STEP 5b apply erosion
    for _ in range(2):
        px_array = computeErosion8Nbh5x5FlatSE(px_array, image_width, image_height)
    

    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('px array')
    axs1[0, 0].imshow(px_array, cmap='gray')

    axs1[0, 1].set_title('threshold')
    axs1[0, 1].imshow(px_array, cmap='gray')

    axs1[1, 0].set_title('strech')
    axs1[1, 0].imshow(contrast_strech, cmap='gray')
    # Compute a dummy bounding box centered in the middle of the input image, and with as size of half of width and height
    # Change these values based on the detected barcode region from your algorithm
    center_x = image_width / 2.0
    center_y = image_height / 2.0
    bbox_min_x = center_x - image_width / 4.0
    bbox_max_x = center_x + image_width / 4.0
    bbox_min_y = center_y - image_height / 4.0
    bbox_max_y = center_y + image_height / 4.0

    # The following code is used to plot the bounding box and generate an output for marking
    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()