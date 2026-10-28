#!/usr/bin/env python3
"""Offline checks: no paid calls, original answers and official denominator preserved."""
import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode = True
import httpx

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".gitignore").is_file())
spec = importlib.util.spec_from_file_location("reextract", ROOT / "evaluations/answer_extraction/reextract_mmlongbench_deepseek.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
RealClient = httpx.AsyncClient


class ReextractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="deepseek_reextract_test_")
        self.directory = Path(self.tmp.name)
        self.rows = [
            {"_sample_index":58,"question":"q58","response":"Appota","answer":"Appota","answer_format":"Str","pred":"Not answerable","score":0,"status":"complete","doc_id":"test.pdf"},
            {"_sample_index":0,"question":"q0","response":"Foo","answer":"Foo","answer_format":"Str","pred":"Foo","score":1,"status":"complete","doc_id":"test.pdf"},
            {"_sample_index":1,"question":"q1","answer":"Not answerable","answer_format":"Str","status":"error","doc_id":"test.pdf"},
            {"_sample_index":2,"question":"q2","response":"A and B","answer":"['A', 'B']","answer_format":"List","pred":"['A', 'B']","score":1,"status":"complete","doc_id":"test.pdf"},
        ]
        self.source = self.directory / "source.json"
        self.source.write_text(json.dumps(self.rows))
        self.args = m.parser().parse_args(["--input",str(self.source),"--output-dir",str(self.directory/"output"),
                                           "--concurrency","2","--attempts","2"])
        self.run = m.ExtractionRun(self.args)
        self.run.root.mkdir()
        self.run.prepare()
        self.calls = []
    def tearDown(self):
        self.tmp.cleanup()
    def handler(self, req):
        body = json.loads(req.content)
        self.calls.append(body)
        assert body["thinking"] == {"type":"enabled"}
        assert body["reasoning_effort"] == "high"
        assert body["stream"] is False
        data = json.loads(body["messages"][-1]["content"].split("Extract from the following data:\n")[-1])
        self.assertEqual(set(data), {"question","model_response"})
        pred = {"q58":"Appota","q0":"Foo","q2":"['A', 'B']"}[data["question"]]
        return httpx.Response(200,json={"id":"SYNTHETIC_ONLY","model":"SYNTHETIC_ONLY",
            "choices":[{"finish_reason":"stop","message":{"content":f"Extracted answer: {pred}\nAnswer format: String","reasoning_content":"TEST_ONLY"}}],
            "usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}})
    def client(self, *a, **kw):
        return RealClient(*a, transport=httpx.MockTransport(self.handler), **kw)
    def test_full_scoring_missing_answer_zero_and_resume(self):
        before=self.source.read_bytes()
        with patch.object(m.httpx,"AsyncClient",self.client):
            summary=asyncio.run(self.run.execute("TEST_ONLY_NOT_SECRET"))
            self.assertEqual(summary["complete"],3)
            self.assertEqual(summary["missing_source_answers"],1)
            self.assertEqual(summary["accuracy_percent"],75)
            self.assertAlmostEqual(summary["f1_percent"],100*6/7)
            self.assertEqual(summary["improved"],1)
            self.assertEqual(summary["worsened"],0)
            self.assertTrue(summary["reextraction_complete"])
            self.assertFalse(summary["evaluation_complete"])
            count=len(self.calls)
            asyncio.run(self.run.execute("TEST_ONLY_NOT_SECRET"))
            self.assertEqual(len(self.calls),count)
        self.assertEqual(self.source.read_bytes(),before)
        merged=m.read(self.run.root/"merged.json")
        self.assertEqual([r.get("response") for r in merged],[r.get("response") for r in self.rows])
        self.assertEqual(merged[2]["score"],0)
        self.assertEqual(merged[0]["original_evaluation"]["pred"],"Not answerable")
        self.assertEqual(len(self.calls),3)
        for path in self.run.root.rglob("*.json"):
            self.assertNotIn("TEST_ONLY_NOT_SECRET",path.read_text())
    def test_no_gt_or_gold_metadata_in_messages(self):
        messages=m.messages_for({"question":"QUESTION","response":"RESPONSE","answer":"GT_SECRET_123",
                                 "answer_format":"GT_FORMAT_SECRET","evidence_pages":"GOLD_PAGE_SECRET"},self.run.prompt)
        text=json.dumps(messages)
        self.assertNotIn("GT_SECRET_123",text)
        self.assertNotIn("GT_FORMAT_SECRET",text)
        self.assertNotIn("GOLD_PAGE_SECRET",text)
    def test_safe_list_and_missing_extraction_format(self):
        with self.assertRaises((ValueError,SyntaxError)):
            m.score_prediction(self.run.scorer,self.rows[3],"[__import__('os').system('false')]")
        with self.assertRaises(ValueError):
            m.parse_prediction("I cannot produce the required format.")
    def test_fatal_auth_stops_before_fanout(self):
        def rejected(req):
            self.calls.append(json.loads(req.content))
            return httpx.Response(402,json={"error":{"message":"SYNTHETIC balance error"}})
        def factory(*a,**kw):
            return RealClient(*a,transport=httpx.MockTransport(rejected),**kw)
        with patch.object(m.httpx,"AsyncClient",factory):
            with self.assertRaises(m.FatalAPIError):
                asyncio.run(self.run.execute("TEST_ONLY_NOT_SECRET"))
        self.assertEqual(len(self.calls),1)
        self.assertEqual(m.read(self.run.root/"merged.summary.json")["status"],"failed")


if __name__=="__main__":
    unittest.main(verbosity=2)
