"""The vendored contract file must stay byte-identical to the canonical one."""

import hashlib
import os
from pathlib import Path


def test_vendored_contract_matches_canonical():
    vendored = (Path(__file__).resolve().parent.parent
                / "src" / "reproagent" / "_vendor" / "env_contract_v1.py")
    resagent_repo = os.environ.get(
        "RESAGENT_REPO",
        str(Path(__file__).resolve().parent.parent.parent / "ResAgent"),
    )
    canonical = Path(resagent_repo) / "contracts" / "env_contract_v1.py"
    assert vendored.is_file(), f"missing vendored file: {vendored}"
    assert canonical.is_file(), f"missing canonical contract: {canonical}"
    v = hashlib.sha256(vendored.read_bytes()).hexdigest()
    c = hashlib.sha256(canonical.read_bytes()).hexdigest()
    assert v == c, (
        "vendored env_contract_v1.py diverged from the canonical contract. "
        "Copy ResAgent/contracts/env_contract_v1.py over — never edit the copy."
    )
