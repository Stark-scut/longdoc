"""Controller status must preserve native tool history and multimodal observations."""
from collections import Counter
import json
from types import SimpleNamespace
import httpx

from test_agent import Base, reply, role_of, tool
from agent.evidence import EvidenceMemory
from agent.loop import validate_history
from agent.status import format_status


def status_of(message):
    content = message.get('content')
    text = content if isinstance(content, str) else '\n'.join(
        part.get('text', '') for part in content or [] if part.get('type') == 'text')
    return text[text.rfind('<agent_status>'):] if '<agent_status>' in text else ''


class StatusContracts(Base):
    def test_initial_budget_and_image_observation_use_no_new_user_messages(self):
        bodies = []
        def handler(request):
            body = json.loads(request.content)
            if role_of(body) == 'locator':
                bodies.append(body)
                step = len(bodies) - 1
                name, args = [('read_page', {'pages': [0]}), ('store', {'pages': [0]}),
                              ('call_reasoner', {})][step]
                return httpx.Response(200, json=reply(tool(name, args, f'id{step}'), 'tool_calls'))
            self.assertNotIn('<agent_status>', json.dumps(body))
            return httpx.Response(200, json=reply({'role': 'assistant', 'content':
                                                 '{"status":"answerable","answer":"red"}'}))
        result, audit = self.run_agent(handler)
        self.assertEqual(result['execution_status'], 'completed')
        initial = status_of(bodies[0]['messages'][0])
        for value in ('Current decision round: 1/6', 'Completed rounds: 0',
                      'Remaining decision rounds (including current): 6', 'Total pages: 2',
                      'Valid page IDs: 0-1', 'Read pages: []', 'Stored pages: []',
                      'Memory version: 0', 'Last result: None', 'Last evaluated memory version: None',
                      'Missing information: None', '- search: 0', '- read_page: 0'):
            self.assertIn(value, initial)
        history = bodies[1]['messages']; validate_history(history)
        read = history[-1]
        self.assertEqual(read['tool_call_id'], 'id0')
        self.assertEqual(read['content'][:2], self.documents['doc'].render_locator([0]))
        self.assertEqual(read['content'][2]['type'], 'text')
        self.assertIn('Current decision round: 2/6', status_of(read))
        self.assertIn('Read but not stored pages: [0]', status_of(read))
        self.assertIn('- read_page: 1', status_of(read))
        self.assertNotIn('OCR_SENTINEL', json.dumps(history))
        for body in bodies:
            self.assertEqual([m['content'] for m in body['messages'] if m['role'] == 'user'], ['What color?'])
        for before, after in zip(bodies, bodies[1:]):
            self.assertEqual(after['messages'][:len(before['messages'])], before['messages'])
        events = [json.loads(line) for line in audit.path.read_text().splitlines()]
        raw = next(e for e in events if e['event'] == 'tool_result' and e['tool_call_id'] == 'id0')
        enhanced = next(e for e in events if e['event'] == 'agent_status' and e['tool_call_id'] == 'id0')
        self.assertFalse(status_of(raw['message']))
        self.assertTrue(status_of(enhanced['message']))
        self.assertEqual(status_of(enhanced['message']), enhanced['text'])

    def test_batch_status_uses_all_outcomes_and_tracks_stale_reasoner_version(self):
        bodies, reasoner_bodies = [], []
        missing = [{'missing_claim': 'Need the second page', 'expected_type': 'text', 'search_hint': 'details'}]
        def handler(request):
            body = json.loads(request.content)
            if role_of(body) == 'locator':
                bodies.append(body)
                turn = len(bodies)
                name, args = [('read_page', {'pages': [0]}), ('store', {'pages': [0]}),
                              ('read_page', {'pages': [1]}), ('store', {'pages': [1]}),
                              ('call_reasoner', {})][turn - 1]
                message = tool(name, args, f'id{turn}')
                if turn == 2:
                    message['tool_calls'] += tool('call_reasoner', {}, 'insufficient')['tool_calls']
                return httpx.Response(200, json=reply(message, 'tool_calls'))
            reasoner_bodies.append(body)
            value = {'status': 'insufficient', 'missing': missing} if len(reasoner_bodies) == 1 else {
                'status': 'answerable', 'answer': 'white'}
            return httpx.Response(200, json=reply({'role': 'assistant', 'content': json.dumps(value)}))
        result, audit = self.run_agent(handler)
        self.assertEqual(result['rounds'], 5)
        batch = bodies[2]['messages']
        self.assertEqual([m['tool_call_id'] for m in batch if m['role'] == 'tool'], ['id1', 'id2', 'insufficient'])
        self.assertFalse(status_of(batch[-2]))
        status = status_of(batch[-1])
        for value in ('Current decision round: 3/6', 'Completed rounds: 2', 'Stored pages: [0]',
                      'Memory version: 1', 'Last result: insufficient', 'Last evaluated memory version: 1',
                      'Need the second page', '- store: 1', '- call_reasoner: 1'):
            self.assertIn(value, status)
        newer = status_of(bodies[4]['messages'][-1])
        self.assertIn('Memory version: 2', newer)
        self.assertIn('Stored pages: [0, 1]', newer)
        self.assertIn('Last evaluated memory version: 1', newer)
        self.assertIn('Read but not stored pages: []', newer)
        self.assertTrue(all('<agent_status>' not in json.dumps(b) for b in reasoner_bodies))
        events = [json.loads(line) for line in audit.path.read_text().splitlines()]
        statuses = [e for e in events if e['event'] == 'agent_status']
        self.assertEqual(len(statuses), 5)
        self.assertTrue(statuses[-1]['terminal'])
        self.assertIn('Current decision round: 5/6', statuses[-1]['text'])
        self.assertIn('Remaining decision rounds (including current): 0', statuses[-1]['text'])
        self.assertIn('Last result: answerable', statuses[-1]['text'])
        for before, after in zip(bodies, bodies[1:]):
            self.assertEqual(after['messages'][:len(before['messages'])], before['messages'])

    def test_last_skipped_tool_includes_primary_error_without_counting_skip(self):
        bodies = []
        def handler(request):
            body = json.loads(request.content)
            if role_of(body) == 'locator':
                bodies.append(body)
                turn = len(bodies)
                if turn == 1:
                    message = tool('store', {'pages': [0]}, 'unread')
                    message['tool_calls'] += tool('call_reasoner', {}, 'skipped')['tool_calls']
                elif turn == 2:
                    message = tool('read_page', {'pages': [0]}, 'read')
                else:
                    message = tool('store', {'pages': [0]}, 'stored')
                    message['tool_calls'] += tool('call_reasoner', {}, 'reason')['tool_calls']
                return httpx.Response(200, json=reply(message, 'tool_calls'))
            return httpx.Response(200, json=reply({'role': 'assistant', 'content':
                                                 '{"status":"answerable","answer":"red"}'}))
        result, _ = self.run_agent(handler)
        self.assertEqual(result['execution_status'], 'completed')
        history = bodies[1]['messages']; validate_history(history)
        self.assertFalse(status_of(history[-2]))
        self.assertEqual(history[-1]['tool_call_id'], 'skipped')
        self.assertIsInstance(history[-1]['content'], str)
        status = status_of(history[-1])
        self.assertIn('invalid_evidence_pages', status)
        self.assertIn('dependency_failed', status)
        self.assertIn('- store: 1', status)
        self.assertIn('- call_reasoner: 0', status)
        self.assertIn('Recent Errors:\n- None', status_of(bodies[2]['messages'][-1]))
        self.assertEqual(result['tool_counts'], {'store': 2, 'read_page': 1, 'call_reasoner': 1})

    def test_sparse_page_ids_and_requested_budget_example(self):
        document = SimpleNamespace(pages={0: None, 2: None, 5: None, 6: None})
        memory = EvidenceMemory(document)
        memory.read_pages.update([0, 2]); memory.store([0])
        tools = SimpleNamespace(counts=Counter(search=5, read_page=4, store=1, call_reasoner=1),
                                last_reason={'status': 'insufficient', 'evidence_version': 1,
                                             'missing': [{'missing_claim': 'Need more evidence'}]})
        config = {**self.config, 'max_rounds': 15}
        status = format_status(memory, tools, config, 11)
        for value in ('Current decision round: 12/15', 'Completed rounds: 11',
                      'Remaining decision rounds (including current): 4',
                      'Total pages: 4', 'Valid page IDs: 0, 2, 5-6', 'Read but not stored pages: [2]'):
            self.assertIn(value, status)

    def test_final_round_has_zero_remaining_without_round_sixteen(self):
        self.config['max_rounds'] = 1
        result, audit = self.run_agent(lambda req: httpx.Response(200, json=reply(
            tool('read_page', {'pages': [0]}, 'read'), 'tool_calls')))
        self.assertEqual(result['stop_reason'], 'max_rounds')
        events = [json.loads(line) for line in audit.path.read_text().splitlines()]
        status = next(e['text'] for e in events if e['event'] == 'agent_status')
        self.assertIn('Current decision round: 1/1', status)
        self.assertIn('Completed rounds: 1', status)
        self.assertIn('Remaining decision rounds (including current): 0', status)
        self.assertNotIn('2/1', status)

    def test_fatal_tool_error_is_logged_in_terminal_status(self):
        turn = 0
        def handler(request):
            nonlocal turn
            body = json.loads(request.content)
            if role_of(body) == 'locator':
                name, args = [('read_page', {'pages': [0]}), ('store', {'pages': [0]}),
                              ('call_reasoner', {})][turn]
                turn += 1
                return httpx.Response(200, json=reply(tool(name, args, f'id{turn}'), 'tool_calls'))
            return httpx.Response(200, json=reply({'role': 'assistant', 'content':
                                                 '{"status":"answerable","answer":["red"]}'}))
        result, audit = self.run_agent(handler)
        self.assertEqual(result['stop_reason'], 'invalid_reasoner_answer')
        events = [json.loads(line) for line in audit.path.read_text().splitlines()]
        status = [e for e in events if e['event'] == 'agent_status'][-1]
        self.assertTrue(status['terminal'])
        self.assertIn('invalid_reasoner_answer', status['text'])
        self.assertIn('Remaining decision rounds (including current): 0', status['text'])
        self.assertEqual(audit.requests['reasoner'], 1)

    def test_plain_response_does_not_fabricate_tool_observations(self):
        result, audit = self.run_agent(lambda req: httpx.Response(200, json=reply(
            {'role': 'assistant', 'content': 'red'})))
        self.assertEqual(result['stop_reason'], 'premature_final')
        self.assertEqual(result['protocol_corrections'], 2)
        events = [json.loads(line) for line in audit.path.read_text().splitlines()]
        self.assertFalse(any(e['event'] in {'tool_result', 'agent_status'} for e in events))
        # Existing protocol-correction user messages are unchanged; status adds none.
        requests = [e['body'] for e in events if e['event'] == 'llm_request']
        self.assertTrue(all(not any(m['role'] == 'tool' for m in b['messages']) for b in requests))
        self.assertTrue(all('<agent_status>' not in m['content'] for b in requests
                            for m in b['messages'] if m['role'] == 'user'))


if __name__ == '__main__':
    import unittest
    unittest.main()
