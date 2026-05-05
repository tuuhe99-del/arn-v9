import tempfile

from fastapi.testclient import TestClient

import arn.api.server as server


def setup_module(module):
    module.tmp = tempfile.TemporaryDirectory()
    server.AUTH_DISABLED = True
    server.pool = server.AgentPool(data_root=module.tmp.name, max_agents=10)
    server.rate_limiter = server.PersistentRateLimiter(f"{module.tmp.name}/rate.db", rpm=1000)
    server.metrics = server.Metrics()


def teardown_module(module):
    server.pool.shutdown_all()
    module.tmp.cleanup()


def test_human_memory_api_endpoints_round_trip():
    c = TestClient(server.app)

    r = c.post('/v1/human/identity/set', json={
        'agent_id': 'apiagent',
        'target_agent': 'developer',
        'name': 'Koda',
        'role': 'coding and tests',
        'must': ['run tests'],
        'must_not': ['claim success without proof'],
    })
    assert r.status_code == 200, r.text

    r = c.post('/v1/human/rule/add', json={
        'agent_id': 'apiagent',
        'target_agent': 'developer',
        'rule': 'Do not claim success unless tests pass',
        'priority': 'critical',
    })
    assert r.status_code == 200, r.text

    r = c.post('/v1/human/procedure/add', json={
        'agent_id': 'apiagent',
        'target_agent': 'developer',
        'name': 'release-audit',
        'steps': ['compile python', 'run pytest'],
        'success': 'tests pass',
    })
    assert r.status_code == 200, r.text

    r = c.post('/v1/human/error/add', json={
        'agent_id': 'apiagent',
        'target_agent': 'developer',
        'mistake': 'CLI handler was missing',
        'fix': 'add handler',
        'lesson': 'smoke test documented commands',
    })
    assert r.status_code == 200, r.text

    r = c.post('/v1/human/context-packet', json={
        'agent_id': 'apiagent',
        'target_agent': 'developer',
        'query': 'audit a release',
        'task': 'audit ARN release',
        'max_tokens': 1000,
    })
    assert r.status_code == 200, r.text
    packet = r.json()['context']
    assert '## Identity' in packet
    assert 'Koda' in packet
    assert 'Do not claim success' in packet
    assert 'release-audit' in packet
    assert 'CLI handler' in packet


def test_memory_store_api_accepts_human_memory_metadata():
    c = TestClient(server.app)
    r = c.post('/v1/memory/store', json={
        'agent_id': 'apiagent2',
        'content': 'Mohamed prefers Python',
        'memory_type': 'preference',
        'scope': 'user',
        'priority': 'high',
        'time_context': 'current',
    })
    assert r.status_code == 200, r.text
    r = c.post('/v1/memory/recall', json={
        'agent_id': 'apiagent2',
        'query': 'what does the user like to code in?',
        'memory_types': ['preference'],
        'top_k': 5,
    })
    assert r.status_code == 200, r.text


def test_cross_agent_share_api_round_trip():
    c = TestClient(server.app)
    r = c.post('/v1/human/share/send', json={
        'agent_id': 'developer',
        'from_agent': 'developer',
        'to_agents': ['manager'],
        'content': 'Researcher confirmed the docs endpoint exists.',
        'task': 'ARN docs',
    })
    assert r.status_code == 200, r.text
    assert r.json()['ok'] is True

    r = c.post('/v1/human/share/inbox', json={
        'agent_id': 'manager',
        'target_agent': 'manager',
        'task': 'ARN docs',
    })
    assert r.status_code == 200, r.text
    rows = r.json()
    assert len(rows) == 1
    assert 'docs endpoint' in rows[0]['content']

    r = c.post('/v1/human/context-packet', json={
        'agent_id': 'manager',
        'target_agent': 'manager',
        'query': 'what did agents share?',
        'task': 'ARN docs',
        'max_tokens': 1000,
    })
    assert r.status_code == 200, r.text
    assert '## Shared Agent Notes' in r.json()['context']
    assert 'docs endpoint' in r.json()['context']
