"""tests/test_serving.py — FastAPI smoke tests (no GPU required)."""
import sys
from pathlib import Path
from unittest.mock import MagicMock,patch
import pytest
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

@pytest.fixture(scope="module")
def client():
    import serving.app as app_mod
    import torch
    mt=MagicMock(); mk=MagicMock()
    mk.eos_token="</s>"; mk.eos_token_id=2
    mk.pad_token="</s>"; mk.pad_token_id=2
    mk.padding_side="left"
    mk.return_value={"input_ids":MagicMock(shape=[1,10])}
    mt.generate.return_value=torch.zeros(1,20,dtype=torch.long)
    mk.decode.return_value="Answer: Net income increased by $0.5B."
    with patch.object(app_mod,"_model",mt),\
         patch.object(app_mod,"_tokenizer",mk),\
         patch.object(app_mod,"_model_dir","mock/r16"):
        from fastapi.testclient import TestClient
        with TestClient(app_mod.app) as c: yield c

class TestHealth:
    def test_200(self,client): assert client.get("/health").status_code==200
    def test_ok(self,client):  assert client.get("/health").json()["status"]=="ok"

class TestRoot:
    def test_200(self,client):    assert client.get("/").status_code==200
    def test_endpoints(self,client): assert "endpoints" in client.get("/").json()

class TestGenerate:
    _p={"question":"What was the revenue change?","context":"Rev 2021:$12B 2020:$10B"}
    def test_200(self,client):    assert client.post("/generate",json=self._p).status_code==200
    def test_schema(self,client):
        b=client.post("/generate",json=self._p).json()
        assert "answer" in b; assert "latency_ms" in b
    def test_no_question(self,client):
        assert client.post("/generate",json={"context":"x"}).status_code==422
    def test_bad_tokens(self,client):
        assert client.post("/generate",json=dict(self._p,max_new_tokens=0)).status_code==422
