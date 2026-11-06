from dataclasses import dataclass,asdict
from agentic.brief import Brief

@dataclass(frozen=True)
class ProviderBrief(Brief):
    provider_name: str = 'Demonstration training provider'
    baseline_course_id: str = ''
    strategy: str = 'Compare new and adapted options'
    delivery: str = 'Applied workshop'

    def validate(self):
        super().validate()
        for key in ['provider_name','baseline_course_id','delivery']:
            value=getattr(self,key)
            if not isinstance(value,str) or len(value)>240:raise ValueError('Invalid provider setting: '+key)
        if not self.provider_name.strip():raise ValueError('Provide a training-provider name')
        if self.strategy not in ['Compare new and adapted options','Adapt an existing course','Design a new course']:
            raise ValueError('Unknown course strategy')
        if self.strategy=='Adapt an existing course' and not self.baseline_course_id:
            raise ValueError('Select the existing course to adapt')
        return self

    def to_dict(self):return asdict(self)
