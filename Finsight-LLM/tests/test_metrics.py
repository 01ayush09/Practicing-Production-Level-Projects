"""tests/test_metrics.py"""
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from eval.metrics import exact_match,token_f1,compute_em_f1,compute_calibration_data,normalize_answer

class TestNormalize:
    def test_lowercase(self): assert normalize_answer("NET") == "net"
    def test_currency(self):   assert normalize_answer("$1,200") == normalize_answer("1200")
    def test_percent(self):    assert normalize_answer("21.5%") == normalize_answer("21.5")

class TestExactMatch:
    def test_identical(self):    assert exact_match("21%","21%")==1.0
    def test_different(self):    assert exact_match("21%","22%")==0.0
    def test_case(self):         assert exact_match("REVENUE","revenue")==1.0

class TestTokenF1:
    def test_perfect(self):  assert token_f1("net income 100","net income 100")==1.0
    def test_zero(self):     assert token_f1("apple","banana")==0.0
    def test_partial(self):  assert 0<token_f1("net income 100","net income 200")<1.0
    def test_both_empty(self): assert token_f1("","")==1.0
    def test_one_empty(self):  assert token_f1("hello","")==0.0

class TestComputeEmF1:
    def test_aggregate(self):
        out=compute_em_f1(["21%","wrong","$2B"],["21%","correct","$2B"])
        assert out["exact_match"] == pytest.approx(2/3,abs=1e-4)
    def test_perfect(self):
        out=compute_em_f1(["a","b"],["a","b"])
        assert out["exact_match"]==1.0; assert out["f1"]==1.0
    def test_mismatch(self):
        with pytest.raises(AssertionError): compute_em_f1(["a"],["b","c"])

class TestCalibration:
    def test_shape(self):
        out=compute_calibration_data([.1,.5,.9],[0,1,1],n_bins=5)
        assert "ece" in out; assert len(out["bin_accuracies"])==5
    def test_ece_range(self):
        import random; random.seed(0)
        c=[random.random() for _ in range(100)]
        k=[float(random.random()>.5) for _ in range(100)]
        assert 0<=compute_calibration_data(c,k)["ece"]<=1.0
