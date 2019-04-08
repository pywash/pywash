class PyWashException(Exception):
    # Base Exception class for the PyWash module
    pass


class FileFormatNotFound(PyWashException):
    """ Raised when a file has an unknown or unparsable format """
    pass