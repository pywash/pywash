from unittest import TestCase
from utils.decorators import strict_types


class TestDecorators(TestCase):
    def test_type_hints_raises(self):
        self.fail()


@strict_types
def type_hint_test(a: int, b: str, c):
    pass
