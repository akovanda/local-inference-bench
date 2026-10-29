#!/usr/bin/env python3
"""Verify deployed API contracts using synthetic prompts; never print credentials."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import httpx
import ollama
from openai import OpenAI


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='https://device.example.invalid:8446')
    parser.add_argument('--token-file', type=Path, default=Path('/external/user/.config/local-general/api-token'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--case', action='append', choices=['metadata','chat','stream','schema','tools','ollama','ollama-tools'])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    key = args.token_file.read_text().strip()
    base = args.base_url.rstrip('/')
    headers = {'Authorization': f'Bearer {key}'}
    client = OpenAI(base_url=base+'/v1', api_key=key, timeout=900, max_retries=0)
    native = ollama.Client(host=base, headers=headers, timeout=900)
    rows: list[dict[str, Any]] = []

    def record(name: str, operation: Any) -> None:
        started = time.monotonic()
        row: dict[str, Any] = {'case': name, 'created_at': datetime.now(timezone.utc).isoformat(), 'benchmark_eligible': False}
        try:
            row['evidence'] = operation()
            row['status'] = 'passed'
        except Exception as exc:
            row['status'] = 'failed'
            # Keep synthetic errors for diagnosis; the token is explicitly removed.
            row['error'] = str(exc).replace(key, '[redacted]')[:3000]
            row['error_type'] = type(exc).__name__
        row['wall_seconds'] = time.monotonic()-started
        rows.append(row)
        (args.output_dir/(name+'.json')).write_text(json.dumps(row, indent=2)+'\n')
        (args.output_dir/'summary.json').write_text(json.dumps({'base_url':base,'benchmark_eligible':False,'cases':rows},indent=2)+'\n')
        print(json.dumps({k:row[k] for k in ('case','status','wall_seconds')}),flush=True)

    def metadata() -> dict[str, Any]:
        with httpx.Client(timeout=15) as h:
            assert h.get(base+'/healthz').status_code == 200
            assert h.get(base+'/readyz').status_code == 200
            docs=h.get(base+'/docs'); assert docs.status_code==200 and 'Local General' in docs.text
            assert h.get(base+'/v1/models').status_code==401
            assert h.post(base+'/v1/responses',headers=headers,json={}).status_code==404
            assert h.get(base+'/api/tags',headers=headers).status_code==200
        names=[m.id for m in client.models.list().data]
        assert 'local-general' in names and 'gpt-oss-20b:64k' in names
        return {'models':names,'docs_public_on_tailnet':True,'generation_requires_key':True}

    def chat() -> dict[str, Any]:
        x=client.chat.completions.create(model='local-general',messages=[{'role':'user','content':'Summarize in one short sentence: Priya owns the deployment. It happens Thursday.'}],max_tokens=2048,temperature=0)
        c=x.choices[0]; text=c.message.content or ''
        assert c.finish_reason=='stop' and 'priya' in text.lower() and 'thursday' in text.lower(), (c.finish_reason,text)
        assert '<think>' not in text and '</think>' not in text
        assert x.model=='local-general'
        return {'answer':text,'finish_reason':c.finish_reason,'usage':x.usage.model_dump() if x.usage else None,'reasoning_separate':bool(getattr(c.message,'reasoning_content',None))}

    def stream() -> dict[str, Any]:
        start=time.monotonic(); first=None; final=''; reasoning_chars=0; reason=None; usage=None; chunks=0
        with client.chat.completions.create(model='local-general',messages=[{'role':'user','content':'Reply with exactly SERVICE_READY.'}],max_tokens=2048,temperature=0,stream=True,stream_options={'include_usage':True}) as events:
            for x in events:
                if first is None:
                    first=time.monotonic()-start
                    with httpx.Client(timeout=5) as h:
                        assert h.get(base+'/readyz').status_code==200, 'readiness failed during active stream'
                chunks+=1
                if x.usage: usage=x.usage.model_dump()
                for c in x.choices:
                    final+=c.delta.content or ''
                    reasoning_chars+=len(getattr(c.delta,'reasoning_content','') or '')
                    reason=c.finish_reason or reason
        assert reason=='stop' and 'SERVICE_READY' in final, (reason,final)
        assert '<think>' not in final and first is not None and chunks>1
        return {'answer':final,'finish_reason':reason,'first_event_seconds':first,'total_seconds':time.monotonic()-start,'chunks':chunks,'reasoning_characters':reasoning_chars,'usage':usage,'ready_during_generation':True}

    def schema() -> dict[str, Any]:
        definition={'type':'object','properties':{'owner':{'type':'string'},'day':{'type':'string'}},'required':['owner','day'],'additionalProperties':False}
        x=client.chat.completions.create(model='local-general',messages=[{'role':'user','content':'Extract owner and day: Priya will deploy on Thursday.'}],max_tokens=2048,temperature=0,response_format={'type':'json_schema','json_schema':{'name':'release','strict':True,'schema':definition}})
        assert x.choices[0].finish_reason=='stop'
        answer=json.loads(x.choices[0].message.content or '')
        assert answer=={'owner':'Priya','day':'Thursday'}, answer
        return {'parsed':answer,'finish_reason':x.choices[0].finish_reason}

    tools_definition=[{'type':'function','function':{'name':'lookup_ticket','description':'Look up a ticket by ID.','parameters':{'type':'object','properties':{'ticket_id':{'type':'string'}},'required':['ticket_id'],'additionalProperties':False}}}]

    def tools() -> dict[str, Any]:
        messages: list[dict[str,Any]]=[{'role':'user','content':'Use lookup_ticket to check ticket INC-42 and report its status.'}]
        x=client.chat.completions.create(model='local-general',messages=messages,tools=tools_definition,tool_choice={'type':'function','function':{'name':'lookup_ticket'}},max_tokens=2048,temperature=0)
        m=x.choices[0].message; assert m.tool_calls and len(m.tool_calls)==1
        call=m.tool_calls[0]; assert call.function.name=='lookup_ticket'
        arguments=json.loads(call.function.arguments); assert arguments=={'ticket_id':'INC-42'},arguments
        messages.append(m.model_dump(exclude_none=True))
        messages.append({'role':'tool','tool_call_id':call.id,'content':json.dumps({'ticket_id':'INC-42','status':'resolved'})})
        y=client.chat.completions.create(model='local-general',messages=messages,tools=tools_definition,tool_choice='none',max_tokens=2048,temperature=0)
        final=y.choices[0].message.content or ''; assert y.choices[0].finish_reason=='stop' and 'resolved' in final.lower(),final
        return {'call_name':call.function.name,'arguments':arguments,'answer':final,'finish_reason':y.choices[0].finish_reason}

    def ollama_chat_generate() -> dict[str, Any]:
        x=native.chat(model='gpt-oss-20b:64k',messages=[{'role':'user','content':'Reply with exactly ALIAS_READY.'}],options={'num_predict':2048,'temperature':0},think=False,stream=False,keep_alive=-1)
        assert x.done and x.model=='local-general' and 'ALIAS_READY' in x.message.content,x.message.content
        assert not getattr(x.message,'thinking',None)
        text=''; end=None; count=0
        for y in native.generate(model='local-general',prompt='Return a JSON object with exactly one field, ready, set to true.',format='json',think=False,stream=True,options={'num_predict':2048,'temperature':0}):
            count+=1;text+=y.response or ''
            if y.done:end=y
        assert end and end.done_reason=='stop' and json.loads(text)=={'ready':True},text
        return {'chat_answer':x.message.content,'alias_resolves_to':x.model,'stream_answer':json.loads(text),'ndjson_events':count,'eval_count':end.eval_count,'done_reason':end.done_reason}

    def ollama_tools() -> dict[str, Any]:
        messages: list[dict[str,Any]]=[{'role':'user','content':'Use lookup_ticket to check ticket INC-42 and report its status. Call the tool before answering.'}]
        x=native.chat(model='local-general',messages=messages,tools=tools_definition,options={'num_predict':2048,'temperature':0},think=False,stream=False)
        assert x.message.tool_calls and len(x.message.tool_calls)==1
        call=x.message.tool_calls[0];assert call.function.name=='lookup_ticket';assert call.function.arguments=={'ticket_id':'INC-42'}
        messages.append(x.message.model_dump(exclude_none=True))
        messages.append({'role':'tool','tool_name':'lookup_ticket','content':json.dumps({'ticket_id':'INC-42','status':'resolved'})})
        y=native.chat(model='local-general',messages=messages,tools=tools_definition,options={'num_predict':2048,'temperature':0},think=False,stream=False)
        answer=y.message.content or '';assert y.done and 'resolved' in answer.lower(),answer
        return {'call_name':call.function.name,'arguments':call.function.arguments,'answer':answer,'done_reason':y.done_reason}

    cases={'metadata':metadata,'chat':chat,'stream':stream,'schema':schema,'tools':tools,'ollama':ollama_chat_generate,'ollama-tools':ollama_tools}
    try:
        for name in args.case or list(cases):record(name,cases[name])
    finally:
        client.close()
    return 1 if any(r['status']=='failed' for r in rows) else 0


if __name__=='__main__':
    raise SystemExit(main())
