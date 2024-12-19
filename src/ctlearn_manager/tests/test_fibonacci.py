import pytest

# copied from https://en.wikipedia.org/wiki/Fibonacci_sequence
FIBONACCI_NUMBERS = (0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144)


@pytest.mark.parametrize(("i", "expected"), enumerate(FIBONACCI_NUMBERS))
def test_fibonacci(i, expected):
    """Test fibonacci function"""
    from template import fibonacci

    assert fibonacci(i) == expected


@pytest.mark.parametrize("i", [-1, -10])
def test_fibonacci_invalid(i):
    """Test fibonacci function raises error on invalid input"""
    from template import fibonacci

    with pytest.raises(ValueError, match=">= 0"):
        fibonacci(i)
