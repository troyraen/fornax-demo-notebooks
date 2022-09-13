# Python Guidelines

## Catching Errors

Take-away:

It is preferable to *let the code fail* if it encounters an error that you haven’t specifically thought through.
Catching errors that you haven’t thought through makes future debugging difficult.

Guidelines:

1.  __*Ideally*__, catch only the specific error(s) that you know can occur and that you know how to make your program properly recover from.
    For example:

    ```python
    try:
        # code that you expect to raise a KeyError sometimes,
        # and you want to handle that case in a specific way

    except KeyError:
        # code to handle the KeyError case
    ```

1.  __*Never do this*__. It’s is called a “bare” except (meaning there's no argument after `except`):

    ```python
    try:
        # code that could throw an error

    except:
        # handle every error that could possibly occur,
        # since the bare except catches *everything*.
    ```

    *Why never?* A bare `except` catches ALL errors that can possibly occur, including system-exiting errors like `SystemExit` and `KeyboardInterrupt`.
    One problem (of many) this can cause:
    if your program gets stuck on the code in the try block, you can’t make it stop without killing the kernel entirely. Using *Control-C* raises a `KeyboardInterrupt` error, but the bare `except` catches it.

1.  __*At a minimum*__, do this.

    ```python
    try:
        # code that could throw an error

    except Exception:
        # handle every non-system-exiting error that could occur
    ```

    *Explanation*:
    `except Exception` catches basically everything other than system-exiting errors.
    So this is preferable to a bare `except` because (for example) *Control-C* will still work as expected.
    This should not be used in-production.
    But it can be useful during development, if you’re having trouble with a bit of code and you want to temporarily move past it before tracking down the specific problem.
    But in general, if you’re tempted to do this, consider not using a `try/except` statement at all -- it’s usually better to just let the error raise.


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

