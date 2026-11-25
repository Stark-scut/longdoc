"""Controller metadata appended to the final real tool observation of a round."""
import json


def page_ranges(pages):
    ranges = []
    for page in sorted(pages):
        if ranges and page == ranges[-1][1] + 1:
            ranges[-1][1] = page
        else:
            ranges.append([page, page])
    return ', '.join(str(start) if start == end else f'{start}-{end}' for start, end in ranges) or 'None'


def format_status(evidence, tools, config, completed_rounds, errors=(), *, terminal=False):
    limit = config['max_rounds']
    remaining = 0 if terminal else max(0, limit - completed_rounds)
    current = completed_rounds + 1 if remaining else completed_rounds
    last = tools.last_reason or {}
    encoded = lambda value: json.dumps(value, ensure_ascii=False)
    lines = [
        '<agent_status>',
        'Round Budget:',
        f'- Current decision round: {current}/{limit}',
        f'- Completed rounds: {completed_rounds}',
        f'- Remaining decision rounds (including current): {remaining}',
        '', 'Document State:',
        f'- Total pages: {len(evidence.document.pages)}',
        f'- Valid page IDs: {page_ranges(evidence.document.pages)}',
        f'- Max pages per read_page call: {config["read_page_max"]}',
        f'- Read pages: {encoded(sorted(evidence.read_pages))}',
        '', 'Evidence Memory:',
        f'- Memory version: {evidence.version}',
        f'- Stored pages: {encoded(sorted(evidence.pages))}',
        f'- Read but not stored pages: {encoded(sorted(evidence.read_pages - evidence.pages))}',
        '', 'Tool Statistics:',
        *(f'- {name}: {tools.counts[name]}' for name in ('search', 'read_page', 'store', 'call_reasoner')),
        '', 'Reasoner Feedback:',
        f'- Last result: {last.get("status", "None")}',
        f'- Last evaluated memory version: {last.get("evidence_version", "None")}',
        f'- Missing information: {encoded(last["missing"]) if "missing" in last else "None"}',
        '', 'Recent Errors:',
        *(f'- {encoded(error)}' for error in errors),
    ]
    if not errors:
        lines.append('- None')
    return '\n'.join([*lines, '</agent_status>'])


def append_status(message, status):
    # Keep image parts and the tool_call_id; never add a second tool observation.
    if isinstance(message['content'], list):
        message['content'].append({'type': 'text', 'text': '\n\n' + status})
    else:
        message['content'] += '\n\n' + status
