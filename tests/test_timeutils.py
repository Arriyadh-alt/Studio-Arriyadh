import pytest

from shorts_clipper.timeutils import format_timestamp, parse_timestamp


@pytest.mark.parametrize("value, expected", [
    ("90", 90), ("1:30", 90), ("01:02:03", 3723), ("0:05.5", 5.5),
])
def test_parse_timestamp(value, expected):
    assert parse_timestamp(value) == expected


@pytest.mark.parametrize("value", ["", "abc", "1:2:3:4", "1:75", "-5", "1::2"])
def test_parse_timestamp_rejects_bad_input(value):
    with pytest.raises(ValueError):
        parse_timestamp(value)


def test_format_timestamp():
    assert format_timestamp(3723.5) == "01:02:03.500"
