"""Parse timestamps like 90, 1:30, 01:02:03 or 1:30.5 into seconds."""


def parse_timestamp(value: str) -> float:
    parts = value.strip().split(":")
    if not 1 <= len(parts) <= 3 or any(p == "" for p in parts):
        raise ValueError(f"Invalid timestamp: {value!r} (use SS, MM:SS or HH:MM:SS)")
    try:
        numbers = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"Invalid timestamp: {value!r} (use SS, MM:SS or HH:MM:SS)") from None
    if any(n < 0 for n in numbers) or any(n >= 60 for n in numbers[1:]):
        raise ValueError(f"Invalid timestamp: {value!r}")
    seconds = 0.0
    for n in numbers:
        seconds = seconds * 60 + n
    return seconds


def format_timestamp(seconds: float) -> str:
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"
