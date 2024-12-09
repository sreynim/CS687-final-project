# returns the argmax (index) of the given array
# if there are multiple, returns one of the best indices uniformly at random
def argmax(arr):
    best_val = np.max(arr)
    best_indices = np.argwhere(arr == best_val)[0]
    return np.random.choice(best_indices)