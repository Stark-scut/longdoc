"""Gold-page input ablation: mocked HTTP, real rendering/extraction/scoring, no GPUs."""
import base64
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

DIRECT = Path(__file__).resolve().parents[1] / 'evaluations/direct_qa'
sys.path.insert(0, str(DIRECT))
import run_answerable
import run_replicas
from evidence_input import EvidenceInput

evaluation = run_replicas.evaluation


class GoldEvidenceInputs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / 'images' / 'doc'
        self.images.mkdir(parents=True)
        self.crops = self.root / 'crops' / 'doc'
        self.crops.mkdir(parents=True)
        self.rows = []
        for i in range(3):
            Image.new('RGB', (64, 64), ['red', 'green', 'blue'][i]).save(self.images / f'page_{i}.png')
            Image.new('RGB', (96, 64), ['yellow', 'black', 'white'][i]).save(self.crops / f'page_{i}.png_b7.jpg')
            self.rows.append({'image_path': f'/old/server/images/doc/page_{i}.png', 'blocks': [
                {'block_id': 7, 'label': 'table', 'content': f'ONLY_PAGE_{i}_OCR'},
                {'block_id': 8, 'label': 'text', 'content': f'PAGE_{i}_WITHOUT_CROP'}]})
        self.blocks = self.root / 'blocks.jsonl'
        self.write_blocks()
        self.paths = (self.images / 'page_0.png', self.images / 'page_2.png')
        self.case = {'doc_id': 'doc.pdf', 'question': 'What is the color?', 'answer': 'red',
                     'answer_format': 'Str', 'evidence_pages': '[1, 3]',
                     'evidence_sources': "['Table']", 'doc_type': 'Academic paper',
                     'input_page_indices': [0, 2]}
        self.input = self.root / 'samples.json'
        self.input.write_text(json.dumps([self.case]))
        self.output = self.root / 'result.json'
        self.model = self.root / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text(json.dumps({'max_position_embeddings': 262144}))

    def write_blocks(self):
        self.blocks.write_text(''.join(json.dumps(row) + '\n' for row in self.rows))

    def args(self):
        return ['--input', str(self.input), '--images-dir', str(self.images.parent),
                '--output', str(self.output), '--evidence-ocr-crops', '--blocks', str(self.blocks),
                '--crop-root', str(self.crops.parent), '--native-resolution', '--max-model-len', '262144']

    def execute(self, args, responses=('red', 'Extracted answer: red\nAnswer format: String')):
        replies = [Mock(ok=True, status_code=200, json=Mock(return_value={
            'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 5}})) for text in responses]
        with patch.object(evaluation.requests, 'Session') as session, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            session.return_value.post.side_effect = replies
            code = evaluation.main(args)
            bodies = [call.kwargs['json'] for call in session.return_value.post.call_args_list]
        return code, bodies

    def test_wire_input_contains_only_gold_pages_and_matches_reasoner_renderer(self):
        provider = EvidenceInput([self.paths], self.blocks, self.crops.parent)
        code, bodies = self.execute(self.args())
        self.assertEqual(code, 0)
        generation, extraction = bodies
        content = generation['messages'][0]['content']
        self.assertEqual(content[0], {'type': 'text', 'text': evaluation.INSTRUCTION})
        self.assertEqual(content[1:-1], provider.content(self.paths))
        self.assertEqual(content[-1]['text'], 'Question: ' + self.case['question'])
        text = '\n'.join(p['text'] for p in content if p['type'] == 'text')
        self.assertIn('ONLY_PAGE_0_OCR', text)
        self.assertIn('ONLY_PAGE_2_OCR', text)
        self.assertNotIn('ONLY_PAGE_1_OCR', text)
        image_urls = [p['image_url']['url'] for p in content if p['type'] == 'image_url']
        expected = [p.read_bytes() for p in provider.image_paths(self.paths)]
        self.assertEqual([base64.b64decode(url.split(',', 1)[1]) for url in image_urls], expected)
        self.assertEqual(len(image_urls), 4)
        self.assertIn('data:image/jpeg;', image_urls[1])
        prompt = evaluation.EXTRACTION_PROMPT_PATH.read_text(encoding='utf-8')
        self.assertEqual(extraction['messages'], [{'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': '\n\nQuestion:What is the color?\nAnalysis:red\n'}])
        for body in bodies:
            self.assertEqual(body['temperature'], 0)
            self.assertEqual(body['seed'], 0)
            self.assertFalse(body['chat_template_kwargs']['enable_thinking'])
        row = json.loads(self.output.read_text())[0]
        self.assertEqual(row['score'], 1)
        self.assertEqual(row['generation_input']['crop_count'], 2)
        self.assertEqual(row['generation_input']['ocr_block_count'], 4)
        self.assertEqual(row['estimated_visual_tokens'], evaluation.native_page_info(provider.image_paths(self.paths))[0])

    def test_baseline_input_stays_image_only_and_uses_same_prompt(self):
        args = self.args()
        args.remove('--evidence-ocr-crops')
        code, bodies = self.execute(args)
        self.assertEqual(code, 0)
        content = bodies[0]['messages'][0]['content']
        self.assertEqual(content[0]['text'], evaluation.INSTRUCTION)
        self.assertEqual(sum(p['type'] == 'image_url' for p in content), 2)
        self.assertNotIn('ONLY_PAGE', json.dumps(content))
        self.assertIn('Page 3:', json.dumps(content))

    def test_rejects_implicit_all_pages_and_truncation(self):
        for mode in ('implicit', 'truncate'):
            with self.subTest(mode=mode):
                case = dict(self.case)
                if mode == 'implicit':
                    case.pop('input_page_indices')
                self.input.write_text(json.dumps([case]))
                args = self.args() + (['--truncate-long-documents'] if mode == 'truncate' else [])
                with self.assertRaises(SystemExit), patch.object(evaluation, 'chat') as chat:
                    with contextlib.redirect_stderr(io.StringIO()):
                        evaluation.main(args)
                chat.assert_not_called()

    def test_missing_selected_ocr_fails_but_missing_crop_is_allowed(self):
        self.rows.pop()
        self.write_blocks()
        with self.assertRaisesRegex(ValueError, 'Missing OCR'):
            EvidenceInput([self.paths], self.blocks, self.crops.parent)
        (self.crops / 'page_0.png_b7.jpg').unlink()
        provider = EvidenceInput([(self.paths[0],)], self.blocks, self.crops.parent)
        self.assertEqual(provider.image_paths((self.paths[0],)), (self.paths[0],))

    def test_large_crops_follow_processor_pixel_cap_without_changing_full_pages(self):
        Image.new('RGB', (640, 640)).save(self.crops / 'page_0.png_b7.jpg')
        provider = EvidenceInput([self.paths], self.blocks, self.crops.parent)
        tokens, largest_page = evaluation.input_image_info(self.paths, provider, 65536)
        self.assertEqual(tokens, 8 + 64 + 6)
        self.assertEqual(largest_page, 4096)
        code, bodies = self.execute(self.args() + ['--dry-run', '--max-pixels', '65536'])
        self.assertEqual(code, 0)
        self.assertEqual(bodies, [])
        self.assertFalse(self.output.exists())

    def test_context_overflow_records_failure_without_dropping_crops(self):
        code, bodies = self.execute(self.args() + ['--max-tokens', '8', '--max-model-len', '27'])
        self.assertEqual(code, 1)
        self.assertEqual(bodies, [])
        row = json.loads(self.output.read_text())[0]
        self.assertEqual(row['status'], 'error')
        self.assertEqual(row['generation_input']['image_count'], 4)
        self.assertNotIn('input_truncation', row)

    def test_resume_rejects_changed_mode_ocr_and_crops(self):
        self.execute(self.args())
        variants = ['mode', 'ocr', 'crop']
        for variant in variants:
            with self.subTest(variant=variant):
                args = self.args()
                original = self.blocks.read_bytes()
                crop = self.crops / 'page_0.png_b7.jpg'
                crop_stat = crop.stat()
                if variant == 'mode':
                    args.remove('--evidence-ocr-crops')
                elif variant == 'ocr':
                    self.blocks.write_bytes(original + b'\n')
                else:
                    import os
                    os.utime(crop, ns=(crop_stat.st_atime_ns, crop_stat.st_mtime_ns + 1000000))
                with self.assertRaises(SystemExit), patch.object(evaluation, 'chat') as chat:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        evaluation.main(args)
                chat.assert_not_called()
                self.blocks.write_bytes(original)

    def test_limit_expansion_preserves_provenance_and_completed_answers(self):
        self.input.write_text(json.dumps([self.case, dict(self.case, question='Second?', input_page_indices=[1])]))
        self.assertEqual(self.execute(self.args() + ['--limit', '1'])[0], 0)
        code, bodies = self.execute(self.args())
        self.assertEqual(code, 0)
        self.assertEqual(len(bodies), 2)
        self.assertIn('ONLY_PAGE_1_OCR', json.dumps(bodies[0]))

    def test_replica_dry_run_forwards_mode_to_every_worker(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = run_replicas.main(['--input', str(self.input), '--images-dir', str(self.images.parent),
                '--model-path', str(self.model), '--output-dir', str(self.root / 'replicas'),
                '--gpu-ids', '0,1', '--no-truncate-long-documents', '--evidence-ocr-crops',
                '--blocks', str(self.blocks), '--crop-root', str(self.crops.parent), '--dry-run'])
        self.assertEqual(code, 0)
        for flag in ('--evidence-ocr-crops', '--blocks ', '--crop-root '):
            self.assertEqual(out.getvalue().count(flag), 2)
        self.assertIn('4 张图片', out.getvalue())
        self.assertFalse((self.root / 'replicas').exists())

    def test_answerable_runner_selects_gold_and_forwards_agent_assets(self):
        agent = self.root / 'agent'
        inputs = agent / 'snapshot/inputs'
        inputs.mkdir(parents=True)
        cases = [dict(self.case, _sample_index=9),
                 dict(self.case, _sample_index=10, answer='Not answerable'),
                 dict(self.case, _sample_index=11, evidence_pages='[]')]
        for case in cases:
            case.pop('input_page_indices')
        (inputs / 'cases.json').write_text(json.dumps(cases))
        (inputs / 'assets.json').write_text(json.dumps({'doc': [{'kind': 'page', 'page': i} for i in range(3)]}))
        (agent / 'run_config.json').write_text(json.dumps({'config': {
            'model_path': str(self.model), 'images_dir': str(self.images.parent),
            'blocks': str(self.blocks), 'crop_root': str(self.crops.parent)}}))
        def replicas(args):
            self.assertIn('--evidence-ocr-crops', args)
            self.assertIn('--no-truncate-long-documents', args)
            self.assertIn(str(self.blocks), args)
            selected = json.loads(Path(args[args.index('--input') + 1]).read_text())
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]['benchmark_sample_index'], 9)
            self.assertEqual(selected[0]['input_page_indices'], [0, 2])
            return 0
        with patch.object(run_replicas, 'main', side_effect=replicas), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_answerable.main(['--gold-evidence-ocr-crops', '--agent-dir', str(agent),
                '--output-dir', str(self.root / 'newrun'), '--dry-run']), 0)
        self.assertFalse((self.root / 'newrun').exists())


if __name__ == '__main__':
    unittest.main()
