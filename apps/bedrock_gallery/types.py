from enum import Enum


class MediaType(Enum):
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"
  
    @classmethod
    def from_string(cls, string_value):
        """Convert string value to corresponding enum member"""
        for member in cls:
            if member.value == string_value:
                return member
        return None