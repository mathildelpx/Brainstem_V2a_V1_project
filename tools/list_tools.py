def get_unique_elements(list1):
    """Return list of unique elements in a given list"""
    list_set = set(list1)
    unique_list = (list(list_set))
    return unique_list


# function to return a list of index corresponding to element in a list (lst) filling a condition
def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]
