class Timespan:
    def __init__(
            self,
            days: int = 0,
            hours: int = 0,
            minutes: int = 0,
            seconds: int = 0,
            milliseconds: int = 0):

        conv_days = days * 3600 * 24 * 1000
        conv_hours = hours * 3600000
        conv_minutes = minutes * 60000
        conv_seconds = seconds * 1000

        self._time = conv_days + conv_hours + conv_minutes + conv_seconds + milliseconds

    @property
    def milliseconds(self) -> int:
        return self._time

    @property
    def seconds(self) -> int:
        return int(self._time / 1000)

    @property
    def minutes(self) -> int:
        return int(self._time / 60000)

    @property
    def hours(self) -> int:
        return int(self._time / 3600000)

    @property
    def days(self) -> int:
        return int(self._time / (3600 * 24 * 1000))

    @classmethod
    def from_days(cls, days: int):
        return cls(days=days)

    @classmethod
    def from_hours(cls, hours: int):
        return cls(hours=hours)

    @classmethod
    def from_minutes(cls, minutes: int):
        return cls(minutes=minutes)

    @classmethod
    def from_seconds(cls, seconds: int):
        return cls(seconds=seconds)

    @classmethod
    def from_milliseconds(cls, milliseconds: int):
        return cls(milliseconds=milliseconds)
