from dataclasses import dataclass
from pathlib import Path
from typing import Self


@dataclass(frozen=True, slots=True)
class Text:
    text: str


@dataclass(frozen=True, slots=True)
class Image:
    path: Path

    def __init__(self, path: str | Path) -> None:
        object.__setattr__(self, "path", Path(path))


@dataclass(frozen=True, slots=True)
class Video:
    path: Path

    def __init__(self, path: str | Path) -> None:
        object.__setattr__(self, "path", Path(path))


@dataclass(frozen=True, slots=True)
class Audio:
    path: Path

    def __init__(self, path: str | Path) -> None:
        object.__setattr__(self, "path", Path(path))


Content = Text | Image | Video | Audio


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: tuple[Content, ...]

    @classmethod
    def user(cls, *content: Content) -> Self:
        return cls(role="user", content=content)

    @classmethod
    def system(cls, *content: Content) -> Self:
        return cls(role="system", content=content)

    @classmethod
    def assistant(cls, *content: Content) -> Self:
        return cls(role="assistant", content=content)
