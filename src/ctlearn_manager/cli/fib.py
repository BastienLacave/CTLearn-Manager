"""Example CLI tool using the example package."""
from argparse import ArgumentParser

from .. import fibonacci
from ..version import __version__

parser = ArgumentParser("fibonacci")
parser.add_argument("n", type=int)
parser.add_argument("--version", action="version", version=__version__)


def main():
    """Compute nth fibonacci number."""
    args = parser.parse_args()
    print(fibonacci(args.n))


if __name__ == "__main__":
    main()
