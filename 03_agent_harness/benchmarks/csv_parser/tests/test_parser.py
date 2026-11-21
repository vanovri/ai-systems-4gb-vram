from parser import parse_csv, parse_csv_line


def test_simple_csv_line() -> None:
    assert parse_csv_line("a,b,c") == ["a", "b", "c"]


def test_quoted_comma_preserved() -> None:
    assert parse_csv_line('"Smith, Jane",42,"New York, NY"') == [
        "Smith, Jane",
        "42",
        "New York, NY",
    ]


def test_escaped_double_quotes() -> None:
    assert parse_csv_line('"He said ""hello""",ok') == ['He said "hello"', "ok"]


def test_empty_fields_and_table() -> None:
    data = 'name,note,score\n"Doe, John","",95\nAlice,,100\n'
    assert parse_csv(data) == [
        ["name", "note", "score"],
        ["Doe, John", "", "95"],
        ["Alice", "", "100"],
    ]
