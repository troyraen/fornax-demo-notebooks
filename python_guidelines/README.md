# Python Guidelines

## Catching Errors

Guidelines:

1.  Catch only the __*specific*__ error(s) that indicate a specific problem -- a problem that you understand and know how to make the program recover from correctly.
    Let any other error raise.

    This example assumes that we expect to get a `ValueError` sometimes, but we know how to fix it:

    ```{code-block} python
    try:
        # code that might raise an error

    # catch only a ValueError
    except ValueError:
        # handle the ValueError
    ```

1.  __*Never*__ use a "bare" `except`.

    ```{code-block} python
    try:
        # code that might raise an error

    # catch ANY AND ALL errors
    except:
        # handle the exception

    ```


## Functions

### Function Arguments

Guideline:

Variables that are used inside of a function should be either:

a. __*passed into*__ the function as an argument, or

b. defined/created __*inside*__ the function.

In other words, functions should not use global variables.

Why:

1. It is easy to forget -- or not even know -- that a function is __*using*__ a global variable, since it is not a function argument.

1. It is easy to loose track of the __*state*__ of a global variable, since it can be changed at almost anytime and in many different ways.

So, using global variables in functions is prone to errors that manifest like:

- "Wait, I just ran this function a minute ago and it worked -- and I haven't changed anything about it or the way I'm calling it. Why isn't it working anymore!?"

- "I'm not sure why it's not working for you... it worked fine on *my* machine.

- "Wait, the function works in *this* part of the code but not in *that* part of the code, but I'm calling it exactly same way both times...?"

