"""
This module contains miscellaneous utility functions.
"""

# ------------------------------------------------------------------------------
# imports

import inspect

# ------------------------------------------------------------------------------

def variable_and_name(variable):
    """Returns the passed variable and the name of the variable as a str."""
    frame = inspect.currentframe().f_back
    variable_name = None
    for name, value in frame.f_locals.items():
        if value is variable:
            variable_name = name
            break
    return variable, variable_name