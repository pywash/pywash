from typing import get_type_hints


def strict_types(func):
    def check_types(*args, **kwargs):
        """ Decorator
        Check if the arguments given to a function are of the correct type for the function.
        If a call to the function supplies the wrong parameters, a TypeError is raised.
        If a call to the function returns the wrong type, a TypeError is raised.
        Also allows partial type hinted functions (Some parameters without hints)
            parameters without type hints don;t get checked and NO error is raised.
        Functions only get executed once

        :raises TypeError: If wrong types are supplied to the function
        :raises TypeError: If wrong output type is returned
        :param args: Given arguments of the function
        :param kwargs: Given keyword arguments of the function
        :return: Result of the function
        """
        hints = get_type_hints(func)
        arguments = kwargs.copy()
        arguments.update(dict(zip(func.__code__.co_varnames, args)))

        for key, value in arguments.items():
            if key in hints:
                if not type(value) == hints[key]:
                    raise TypeError("Type of {} is {} and not {}.".
                                    format(key, type(value), hints[key]))

        # Call the function to test the result
        result = func(*args, **kwargs)

        # Check if the result has the same type as the type hint (if any)
        if 'return' in hints:
            if not type(result) == hints['return']:
                raise TypeError("Type of result is {} and not {}.".format(type(result), hints['return']))
        return result
    return check_types
