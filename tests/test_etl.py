from src.etl_build_dataset import choose_engine
from pathlib import Path

def test_choose_engine_xlsx():
    assert choose_engine(Path("test.xlsx")) == "openpyxl"

def test_choose_engine_xls():
    assert choose_engine(Path("test.xls")) == "xlrd"
