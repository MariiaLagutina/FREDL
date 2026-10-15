"""Hash-pinned human inputs and synthetic end-to-end collection/evaluation."""

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pytest

from src.__main__ import main
from src.cli import CliError
from src.evaluation.pdf.multibook.scoring.cli import (
    pdf_collect_multibook_judgments, pdf_evaluate_multibook_retrieval,
)
from src.evaluation.pdf.multibook.scoring.judgment_models import (
    PdfCollectedJudgments,
)
from src.evaluation.pdf.multibook.scoring.metric_models import (
    PdfMultibookRetrievalEvaluation,
)
from src.evaluation.pdf.multibook.pooling.cli import (
    pdf_pool_multibook_judgments,
)
from src.evaluation.pdf.multibook.pooling.models import (
    PdfRetrievalJudgmentPool,
)
from tests.evaluation.pdf.multibook.pooling import pool_helpers as pool_helpers
from tests.evaluation.pdf.multibook.retrieval import runner_helpers as runner
from tests.evaluation.pdf.multibook.scoring.metric_helpers import fill


pdf_inputs = runner.pdf_inputs
package_inputs = pool_helpers.package_inputs
forbid_real_model_loading = runner.forbid_real_model_loading


@pytest.fixture()
def evaluation_inputs(
    package_inputs: tuple[Path, Path, Path], tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    root, golden, runs = package_inputs
    review = tmp_path / 'pool'
    pdf_pool_multibook_judgments(
        str(golden), str(root), str(runs), str(review),
    )
    pool = review / 'judgment-pool.json'
    pool_data = json.loads(pool.read_bytes())
    fingerprint = PdfRetrievalJudgmentPool.model_validate_json(
        json.dumps(pool_data),
    ).fingerprint
    sheet = tmp_path / 'human.csv'
    sheet.write_text(fill((review / 'worksheet.csv').read_text(),
                          lambda row: ('irrelevant', '')))
    manifest = tmp_path / 'inputs.json'
    manifest.write_text(json.dumps({
        'pool': {'path': str(pool),
                 'sha256': sha256(pool.read_bytes()).hexdigest()},
        'pool_fingerprint': fingerprint,
        'worksheets': [{'path': sheet.name,
                        'sha256': sha256(sheet.read_bytes()).hexdigest(),
                        'reviewer': 'Synthetic reviewer'}],
    }))
    return root, golden, runs, manifest


def test_public_commands_are_read_only_and_write_versioned_artifacts(
    evaluation_inputs: tuple[Path, Path, Path, Path], tmp_path: Path,
    capsys: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, golden, runs, manifest = evaluation_inputs
    sheet = tmp_path / 'human.csv'
    before = {path: path.read_bytes()
              for path in (sheet, manifest, golden, runs)}

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail('Evaluation may not execute retrieval')
    monkeypatch.setattr(
        'src.evaluation.pdf.bm25_baseline.PdfBm25Baseline.search', forbidden,
    )
    collected = tmp_path / 'collected.json'
    output = tmp_path / 'metrics.json'
    for command, target in [('pdf_collect_multibook_judgments', collected),
                            ('pdf_evaluate_multibook_retrieval', output)]:
        main([command, str(golden), str(root), str(runs), str(manifest),
              str(target)])
    labels = PdfCollectedJudgments.model_validate_json(collected.read_bytes())
    evaluation = PdfMultibookRetrievalEvaluation.model_validate_json(
        output.read_bytes(),
    )
    assert evaluation.judgments_fingerprint == labels.fingerprint
    assert evaluation.artifact_hashes['pool'] == json.loads(
        manifest.read_bytes(),
    )['pool']['sha256']
    assert evaluation.runs[0].artifact_sha256
    assert before == {path: path.read_bytes() for path in before}
    assert 'fingerprint' in capsys.readouterr().out
    assert not list(tmp_path.glob('.fredl-eval-*'))
    with pytest.raises(CliError, match='new file'):
        pdf_evaluate_multibook_retrieval(
            str(golden), str(root), str(runs), str(manifest), str(output),
        )


@pytest.mark.parametrize('drift', [
    'sheet_hash', 'sheet_bytes', 'pool_hash', 'pool_fingerprint',
    'blank', 'missing_row',
    'unknown', 'duplicate_path', 'golden_bytes', 'query_mode',
])
def test_incomplete_or_drifted_inputs_cannot_publish_final_metrics(
    evaluation_inputs: tuple[Path, Path, Path, Path], tmp_path: Path,
    drift: str,
) -> None:
    root, golden, runs, manifest = evaluation_inputs
    data = json.loads(manifest.read_bytes())
    sheet = tmp_path / 'human.csv'
    if drift == 'sheet_hash':
        data['worksheets'][0]['sha256'] = '0' * 64
    elif drift == 'sheet_bytes':
        sheet.write_bytes(sheet.read_bytes() + b'\n')
    elif drift == 'pool_hash':
        data['pool']['sha256'] = '0' * 64
    elif drift == 'pool_fingerprint':
        data['pool_fingerprint'] = '0' * 64
    elif drift == 'duplicate_path':
        data['worksheets'].append(data['worksheets'][0])
    elif drift == 'golden_bytes':
        golden.write_bytes(golden.read_bytes() + b'\n')
    elif drift == 'query_mode':
        pool_path = Path(data['pool']['path'])
        value = json.loads(pool_path.read_bytes())
        value['query_plan']['mode'] = 'original'
        pool_path.write_text(json.dumps(value))
        data['pool']['sha256'] = sha256(pool_path.read_bytes()).hexdigest()
    else:
        text = sheet.read_text()
        if drift == 'blank':
            text = text.replace('irrelevant', '')
        elif drift == 'missing_row':
            text = '\n'.join(text.splitlines()[:-1]) + '\n'
        else:
            text = text.replace('W0001', 'Unknown')
        sheet.write_text(text)
        data['worksheets'][0]['sha256'] = sha256(
            sheet.read_bytes(),
        ).hexdigest()
    manifest.write_text(json.dumps(data))
    output = tmp_path / 'metrics.json'
    with pytest.raises(CliError) as error:
        pdf_evaluate_multibook_retrieval(
            str(golden), str(root), str(runs), str(manifest), str(output),
        )
    if drift == 'blank':
        assert 'Incomplete judgments:' in str(error.value)
        assert 'queues' in str(error.value)
        # A collector draft is allowed to retain blank labels.
        pdf_collect_multibook_judgments(
            str(golden), str(root), str(runs), str(manifest),
            str(tmp_path / 'draft.json'),
        )
    assert not output.exists()
    assert not list(tmp_path.glob('.fredl-eval-*'))


@pytest.mark.parametrize('failure', ['unsupported', 'race'])
def test_failed_output_publication_never_overwrites_or_leaves_temporary_files(
    evaluation_inputs: tuple[Path, Path, Path, Path], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    root, golden, runs, manifest = evaluation_inputs
    output = tmp_path / 'metrics.json'

    def fail_link(source: Path, target: Path) -> None:
        if failure == 'race':
            target.write_bytes(b'other writer')
        raise OSError('Synthetic publication failure')
    monkeypatch.setattr(
        'src.evaluation.pdf.multibook.scoring.cli.os.link', fail_link,
    )
    with pytest.raises(CliError):
        pdf_evaluate_multibook_retrieval(
            str(golden), str(root), str(runs), str(manifest), str(output),
        )
    if failure == 'race':
        assert output.read_bytes() == b'other writer'
    else:
        assert not output.exists()
    assert not list(tmp_path.glob('.fredl-eval-*'))
