class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):
    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array

def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
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

def find_edges(pixel_array, image_width, image_height):
    output = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if 0 <= i+di < image_height and 0 <= j+dj < image_width:
                        if pixel_array[i + di][j + dj] != pixel_array[i][j]:
                            # pixel misses
                            output[i][j] = pixel_array[i + di][j + dj]
                            break
                
    return output



def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
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


def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    map = [(0, -1), (-1, 0), (1, 0), (0, 1)]

    # initialise label counter and dictionary to keep track of label sizes
    c = 1
    label_dict = {}
    
    # create an output image array and visited flag array with same size as input image
    labeled_image = createInitializedGreyscalePixelArray(image_width, image_height)
    visited = createInitializedGreyscalePixelArray(image_width, image_height)
    
    queue = Queue()

    for i in range(image_height):
        for j in range(image_width):

            if pixel_array[i][j] > 0 and visited[i][j] == 0:
                queue.enqueue((i, j))
                while not queue.isEmpty():
                    x, y = queue.dequeue()

                    if visited[x][y] == 0:
                        visited[x][y],labeled_image[x][y] = 1,c

                        dic = label_dict.get(c,None)
                        if dic == None:
                            label_dict[c] = [1, x, x, y, y]
                            
                        else:
                            dic = label_dict[c]
                            x_min = min(dic[1], x)
                            x_max = max(dic[2], x)
                            y_min = min(dic[3], y)
                            y_max = max(dic[4], y)
                            count = dic[0] + 1
                            label_dict[c] = [count, x_min, x_max, y_min, y_max]

                        for dx, dy in map:

                            if 0 <= x+dx < image_height and 0 <= y+dy < image_width:
                                if pixel_array[x+dx][y+dy] > 0 and visited[x+dx][y+dy] == 0:
                                    queue.enqueue((x+dx, y+dy))
                c += 1

    return labeled_image, label_dict




image_width = 16
image_height = 16
pixel_array = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
(ccimg,ccsizes) = computeConnectedComponentLabeling(pixel_array,image_width,image_height)

# {1: 19, 2: 59}

print(ccsizes)