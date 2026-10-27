from feelings import __version__
from feelings.main import greet, process_items


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_greet() -> None:
    assert greet("RC") == "Hello, RC!"


def test_process_items_empty() -> None:
    assert process_items([])["count"] == 0


def test_process_items() -> None:
    stats = process_items([1, 2, 3])
    assert stats["sum"] == 6
    assert stats["mean"] == 2.0
