"""tests/test_data_pipeline.py"""
import json,os,sys,tempfile
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from data.prepare_dataset import (build_answer,build_context,format_alpaca,
                                   _synthetic_fallback,load_and_process,write_jsonl)
from data.dataset_utils import build_prompt,load_jsonl

class TestBuildContext:
    def test_all_fields(self):
        ex={"pre_text":["Revenue grew."],"table":[["","2021"],["Rev","$10B"]],"post_text":["Commentary."]}
        ctx=build_context(ex)
        assert "Revenue grew" in ctx; assert "2021" in ctx
    def test_empty(self):
        assert "No additional context" in build_context({})
    def test_table_pipe(self):
        assert "|" in build_context({"table":[["A","B"],["1","2"]]})

class TestBuildAnswer:
    def test_with_steps(self):
        ex={"answer":"21%","steps":[{"op":"divide","args":["a","b"],"res":"0.21"}]}
        a=build_answer(ex); assert "Step 1" in a; assert "21%" in a
    def test_without_steps(self):
        assert "42%" in build_answer({"answer":"42%"})

class TestFormatAlpaca:
    _ex={"id":"t1","question":"Net income?","pre_text":["Q4 strong."],
         "table":[["","2021"],["NI","$2B"]],"post_text":[],"answer":"$2B","steps":[]}
    def test_returns_dict(self): assert isinstance(format_alpaca(self._ex),dict)
    def test_keys(self):
        r=format_alpaca(self._ex)
        for k in ("instruction","input","output","question","answer"): assert k in r
    def test_none_on_empty(self):
        assert format_alpaca({"question":"","answer":""})==None

class TestWriteJsonl:
    def test_roundtrip(self):
        records=[{"a":1},{"b":2}]
        with tempfile.NamedTemporaryFile(suffix=".jsonl",delete=False) as f: path=f.name
        try: write_jsonl(records,path); assert load_jsonl(path)==records
        finally: os.unlink(path)

class TestSyntheticFallback:
    def test_splits(self):
        ds=_synthetic_fallback()
        assert "train" in ds; assert len(ds["train"])>0
    def test_fields(self):
        row=_synthetic_fallback()["train"][0]
        for f in ("id","question","answer","table"): assert f in row

class TestBuildPrompt:
    _ex={"instruction":"Answer.\n\nQuestion: Net income?","input":"NI: $2B","output":"Answer: $2B"}
    def test_includes_output(self): assert "Answer: $2B" in build_prompt(self._ex,True)
    def test_excludes_output(self): assert "Answer: $2B" not in build_prompt(self._ex,False)
