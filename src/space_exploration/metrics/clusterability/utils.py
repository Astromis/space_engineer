import numpy as np

def getRandomArray(arr, m):
    '''
        input: arr: an input array in d dimensions
               m: the number of random points generated
        return: an array of m random points in the same dimension as the input array
    '''

    # The list of minimum values and the list of maximum values
    # This two lists define the boundary of the area for randomly generating samples
    # We assume the input array has different scales along its dimensions
    mins = []
    maxs = []

    dims = arr.shape[1]

    for i in range(dims):
        mins.append(arr[:, i].min())
        maxs.append(arr[:, i].max())

    ans = np.zeros((m, dims))

    for i in range(m):
        ans[i] = np.random.uniform(mins, maxs)

    return ans