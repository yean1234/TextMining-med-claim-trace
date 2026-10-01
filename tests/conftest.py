from pathlib import Path

import pytest

from medclaim.coding.features import JournalMatcher
from medclaim.coding.strength import EnglishClaimCoder, KoreanClaimCoder
from medclaim.nlp.lexicon import Lexicon
from medclaim.paths import DEFAULT_CONFIG_DIR
from medclaim.utils import load_yaml

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def lex_ko():
    return Lexicon.load(DEFAULT_CONFIG_DIR / "lexicon_ko.yaml")


@pytest.fixture(scope="session")
def lex_en():
    return Lexicon.load(DEFAULT_CONFIG_DIR / "lexicon_en.yaml")


@pytest.fixture(scope="session")
def ko(lex_ko):
    return KoreanClaimCoder(lex_ko)


@pytest.fixture(scope="session")
def en(lex_en):
    return EnglishClaimCoder(lex_en)


@pytest.fixture(scope="session")
def journals():
    return JournalMatcher(load_yaml(DEFAULT_CONFIG_DIR / "journals.yaml"))
