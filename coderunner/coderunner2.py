import math



def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0.0):
    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array

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



def computeMedian5x3ZeroPadding2(pixel_array, image_width, image_height):
    result_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for x in range(image_height):
        for y in range(image_width):
            pixels = []
            for i in range(-1, 2):
                for j in range(-2, 3):
                    new_x = x + i
                    new_y = y + j
                    if (new_x < 0 or new_x >= image_height or new_y < 0 or new_y >= image_width):
                        pixels.append(0)
                    else:
                        pixels.append(pixel_array[new_x][new_y])
            pixels.sort()
            result_pixel_array[x][y] = pixels[len(pixels) // 2]
    return result_pixel_array

def computeMedian5x3ZeroPadding(pixel_array, image_width, image_height):
    output_image = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            pixel_values = []

            for di in range(-1, 2):
                for dj in range(-2, 3): 
                    if (i+di < 0 or i+di >= image_height or j+dj < 0 or j+dj >= image_width):
                        pixel_values.append(0)
                    else:
                        pixel_values.append(pixel_array[i+di][j+dj])

            pixel_values.sort()
            output_image[i][j] = pixel_values[len(pixel_values) // 2]
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