"""最小冒烟测试：确认 RDKit 可用、目录创建正常。"""

from molpredict import __version__
from molpredict.config import MODELS_DIR, PROCESSED_DIR, RAW_DIR, ensure_dirs


def test_version():
    assert __version__


def test_rdkit_available():
    from rdkit import Chem

    mol = Chem.MolFromSmiles("c1ccccc1")
    assert mol is not None
    assert Chem.MolToInchiKey(mol).startswith("UHOVQNZJYSORNB")


def test_ensure_dirs():
    ensure_dirs()
    assert RAW_DIR.is_dir()
    assert PROCESSED_DIR.is_dir()
    assert MODELS_DIR.is_dir()
