from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Brief:
    sector: str
    audience: str
    business_need: str
    level_min: int = 2
    level_max: int = 3
    hours: int = 24
    scenario: str = "expanded"

    def validate(self):
        if not isinstance(self.sector, str) or not self.sector.strip() or len(self.sector)>120:
            raise ValueError("Provide a sector of at most one hundred twenty characters")
        if not isinstance(self.audience, str) or not 4<=len(self.audience.strip())<=300:
            raise ValueError("Provide a specific target audience")
        if not isinstance(self.business_need, str) or not 15<=len(self.business_need.strip())<=1200:
            raise ValueError("Provide a concrete business need")
        if type(self.level_min) is not int or type(self.level_max) is not int or not 1<=self.level_min<=self.level_max<=5:
            raise ValueError("Choose an AI-depth interval between one and five")
        if type(self.hours) is not int or not 8<=self.hours<=120:
            raise ValueError("Proposed course duration must be between eight and one hundred twenty hours")
        if self.scenario not in ["legacy", "strict", "expanded", "student"]:
            raise ValueError("Unknown evidence scenario")
        return self

    def to_dict(self):
        return asdict(self)
