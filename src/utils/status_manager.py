from enum import Enum


class Status(Enum):
    OK = 200
    PARTIAL = 206
    NOT_FOUND = 404
    INVALID = 403
