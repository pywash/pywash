from unittest import TestCase
from src.utils.decorators import strict_types


class TestDecorators(TestCase):
    def test_argument_raises(self):
        """
        Test that checks if type_hint decorated functions raises TypeErrors
         when arguments given are not of the correct type
        :return: Test Passed
        """
        with self.assertRaises(TypeError):
            # These functions should raise TypeErrors
            type_hint_test(1, 1, 1)
            type_hint_test('a', 'a', 'a')

    def test_return_raises(self):
        """
        Test that checks if type_hint decorated functions raises TypeErrors
         when return statements are not of the correct type
        :return: Test Passed
        """
        with self.assertRaises(TypeError):
            # Thi function should raise TypeErrors
            type_hint_test(1, 'x', 'x')

    def test_type_hints_not_raises(self):
        """
        Test that checks if type_hint decorated functions
         behave the same when all type requirements are met
        :return: Test Passed
        """
        x = 1
        self.assertEqual(type_hint_test(x, 'x', x), x, msg="Type hint didn't raise an error.")


@strict_types
def type_hint_test(a: int, b: str, c) -> int:
    """ Decorated Test Function """
    return c
