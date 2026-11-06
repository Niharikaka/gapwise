"""DeepSeek transport with a separate cumulative agent-testing budget."""
from __future__ import annotations
from datetime import datetime, timezone
try:
    import fcntl
except ImportError:  # Windows: the in-process lock still serialises writes
    fcntl = None
import json
import os
from pathlib import Path
import threading
import uuid
import urllib.request
import urllib.error

from data import OUTPUT
from deepseek_client import PRICES, credential

AGENT_OUTPUT = OUTPUT/'agentic'
_LOCK = threading.RLock()


class BudgetExhausted(RuntimeError):
    pass


class AgentBudget:
    def __init__(self, limit, directory=AGENT_OUTPUT):
        if not isinstance(limit,(int,float)) or not 0<limit<=100:
            raise ValueError('Set an explicit positive cumulative agent budget, at most one hundred dollars')
        self.limit, self.directory = float(limit), Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.path = self.directory/'api_ledger.jsonl'

    def _event(self, amount, event, request_id, maximum=False):
        with _LOCK, (self.directory/'.budget.lock').open('a') as lock:
            if fcntl: fcntl.flock(lock,fcntl.LOCK_EX)
            spent = sum(json.loads(x)['amount_usd'] for x in self.path.read_text().splitlines()) if self.path.exists() else 0.0
            if maximum and spent+amount>self.limit:
                raise BudgetExhausted('The cumulative agent API allowance has been reached')
            row = {'event':event,'request_id':request_id,'amount_usd':amount,
                   'at':datetime.now(timezone.utc).isoformat()}
            with self.path.open('a') as stream:
                stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
            return spent+amount

    def reserve(self, request_id, amount):
        return self._event(amount,'reserve',request_id,maximum=True)

    def settle(self, request_id, reserved, actual):
        return self._event(actual-reserved,'settle',request_id)

    def spent(self):
        return sum(json.loads(x)['amount_usd'] for x in self.path.read_text().splitlines()) if self.path.exists() else 0.0


class DeepSeekProvider:
    is_live = True

    def __init__(self, budget_usd, model='deepseek-flash',max_tokens=3000):
        if model not in PRICES:
            raise ValueError('Unsupported model/pricing configuration')
        if type(max_tokens) is not int or not 512<=max_tokens<=6000:raise ValueError('Invalid maximum output size')
        self.max_tokens=max_tokens
        self.model = model
        self.budget = AgentBudget(budget_usd)

    def complete(self, messages, tools=None, tool_choice=None, json_mode=False):
        body = {'model':self.model,'messages':messages,'thinking':{'type':'disabled'},
                'max_tokens':self.max_tokens,'temperature':0.2}
        if tools:
            body['tools']=tools
            body['tool_choice']=tool_choice or 'auto'
        if json_mode:
            body['response_format']={'type':'json_object'}
        token_bound = len(json.dumps(body,ensure_ascii=False).encode())+512
        inp,out = PRICES[self.model]
        reservation = (token_bound*inp+self.max_tokens*out)/1e6
        request_id = uuid.uuid4().hex
        # Check credential before reserving funds; configuration errors spend nothing.
        key = credential()
        self.budget.reserve(request_id,reservation)
        req = urllib.request.Request('https://api.deepseek.com/chat/completions',data=json.dumps(body).encode(),
              headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=110) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'DeepSeek HTTP {exc.code}; the request reservation is retained conservatively') from None
        usage = result['usage']
        actual = (usage['prompt_tokens']*inp+usage['completion_tokens']*out)/1e6
        self.budget.settle(request_id,reservation,actual)
        choice = result['choices'][0]
        if choice['finish_reason'] not in ['stop','tool_calls']:
            raise RuntimeError('The model response was incomplete')
        return {'message':choice['message'],'usage':usage,'model':result['model'],
                'cost_upper_usd':actual,'request_id':request_id}
