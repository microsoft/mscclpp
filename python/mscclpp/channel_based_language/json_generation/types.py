from mscclpp.channel_based_language.types import BufferType
from enum import Enum
from dataclasses import dataclass

class InfoLocation(Enum):
    gpu = "gpu"
    cpu = "cpu"

@dataclass
class RemoteBuffer:
    rank: int
    type: BufferType
    info_location: InfoLocation

    def convert_to_json(self):
        return {
            "rank": self.rank,
            "type": str(self.type),
            "info_location": str(self.info_location)
        }
    
    def __hash__(self):
        return hash((self.rank, self.type, self.info_location))