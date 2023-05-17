# Q1
def computeHistogram(pixel_array, image_width, image_height, nr_bins = 8):
    '''computes the histogram of the pixel array'''
    histogram = [0] * nr_bins
    for i in range(len(pixel_array)):
        for j in range(len(pixel_array[0])):
            histogram[int(pixel_array[i][j])] += 1
    return histogram

# Q2 (include Q1 code)
def computeCumulativeHistogram(pixel_array, image_width, image_height, nr_bins):
    '''computes the cumulative histogram of the pixel array'''

    histogram = computeHistogram(pixel_array, image_width, image_height, nr_bins)

    cumulative_histogram = [0] * nr_bins
    cumulative_histogram[0] = histogram[0]
    for i in range(len(cumulative_histogram)):
        cumulative_histogram[i] = cumulative_histogram[i - 1] + histogram[i]
    return cumulative_histogram

#Q3
def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    '''computes the thresholded image'''
    return list(map(lambda x: list(map(lambda y: 255 if y >= threshold_value else 0, x)), pixel_array))


#Q4
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):
    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array

#Q4
def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):
            greyscale_pixel_array[i][j] = int(round(0.299 * pixel_array_r[i][j] + 0.587 * pixel_array_g[i][j] + 0.114 * pixel_array_b[i][j],0))

    return greyscale_pixel_array

#Q5
def computeMinAndMaxValues(pixel_array, image_width, image_height):
    '''computes the minimum and maximum values of the pixel array'''
    flat_array = [pixel_array[i][j] for i in range(image_height) for j in range(image_width)]
    return (min(flat_array), max(flat_array))

#Q5
def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    '''scales the greyscale pixel array to 0-255 and quantizes it to 8 bits'''
    min, max = computeMinAndMaxValues(pixel_array, image_width, image_height)
    if min-max == 0:
        return list(map(lambda x: list(map(lambda y: 0, x)), pixel_array))
    return list(map(lambda x: list(map(lambda y: int(round(((y - min) * 255 / (max - min)),0)), x)), pixel_array))

#Q6
def computeHistogramArbitraryNrBins(pixel_array, image_width, image_height, nr_bins):
    '''computes the histogram of the pixel array with arbitrary number of bins'''
    histogram = [0] * nr_bins

def computeHistogramArbitraryNrBins(pixel_array, image_width, image_height, nr_bins):
    histogram = [0.0] * nr_bins
    bin_size = round((256/nr_bins),1)

    for i in range(image_height):
        for j in range(image_width):
            bin_index = int(pixel_array[i][j] / bin_size)
            histogram[bin_index] += 1

    return histogram
