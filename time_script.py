# Benchmark different methods using timeit
import timeit

import pandas as pd

# Define test cases
empty_nested_list = [[]]  # l1 == [[]]
empty_list = []  # l1 == []

# Predefine object for `is` comparison
predefined_empty = [[]]


# Define test functions
def direct_comparison():
    return empty_nested_list == [[]]


def length_empty_check():
    return len(empty_nested_list) == 1 and not empty_nested_list[0]


def identity_check():
    return empty_nested_list is predefined_empty


def repr_comparison():
    return repr(empty_nested_list) == "[[]]"


def id_comparison():
    return id(empty_nested_list) == id([[]])


def tuple_conversion():
    return tuple(map(tuple, empty_nested_list)) == ((),)


def index_check():
    return empty_nested_list[0] == []


def safe_index_check():
    return empty_nested_list and empty_nested_list[0] == []


# Run timing tests
methods = {
    "Direct Comparison (==)": direct_comparison,
    "Length + Empty Check": length_empty_check,
    "Identity Check (`is`)": identity_check,
    "repr() Comparison": repr_comparison,
    "id() Comparison": id_comparison,
    "Tuple Conversion": tuple_conversion,
    "Index Check (Unsafe)": index_check,
    "Safe Index Check": safe_index_check,
}

# Time each method
results = {
    name: timeit.timeit(method, number=1_000_000) for name, method in methods.items()
}

# Display results in a table
df = pd.DataFrame(results.items(), columns=["Method", "Time (seconds)"]).sort_values(
    by="Time (seconds)"
)
print("Optimized Benchmark of Methods:")
print(df)
