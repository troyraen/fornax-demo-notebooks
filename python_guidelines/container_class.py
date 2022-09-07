# Create a simple class and use it as a container for a set of related parameters.

# A named tuple  's are perfect when you want a tuple that let's you access the elements by name.
#  like: `mytuple.`

# This is a great way to create a container for a set of parameters.

# The example class created below is called `MyParams`.
# It is defined using the base-class `typing.NamedTuple`.
# Thus, an instance of MyParams will be a regular tuple with the added feature that you can access the tuple elements by name (rather than only by index)
# This makes them convenient for storing, passing around, and a set of parameters .
# and for accessing
# Benefits:
# - pass the full set of parameters to a function by sending a single argument, rather than passing separate arguments for every parameter.
# - access an individual parameter by name

from typing import NamedTuple


# dummy data for the parameters that we want to group together in the container
# this example uses 3 parameters with types int, float, and str respectively
myparam1 = 329
myparam2 = 0.33288
myparam3 = "my param value"


# create the class
# this is where we define the names for each tuple element
class MyParams(NamedTuple):
    """Container for a set of parameters.

    Attributes
    ----------
    myparam1: int
        <description of this parameter>
    myparam2: float
        <description of this parameter>
    myparam3: str
        <description of this parameter>
    """
    myparam1: int
    myparam2: float
    myparam3: str


# use the class
# first, create an instance using the dummy data
my_params = MyParams(myparam1, myparam2, myparam3)

# the parameters can now be accessed by name, using the dot notation
my_params.myparam1  # this is the value 329
