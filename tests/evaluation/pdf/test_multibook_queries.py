"""Original and clarified query-plan contracts."""

import json

from pydantic import ValidationError
import pytest

from src.evaluation.pdf.library import PdfLibraryCorpus
from src.evaluation.pdf.multibook_golden import PdfMultibookGolden
from src.evaluation.pdf.multibook_queries import (
    PdfMultibookQueryPlan,
    PdfQueryMode,
    build_pdf_multibook_query_plan,
    require_pdf_query_plan_matches,
)

from tests.evaluation.pdf import (
    multibook_contract_helpers as contract_fixtures,
)
from tests.evaluation.pdf.multibook_contract_helpers import _manifest
from tests.evaluation.pdf.multibook_golden_helpers import _golden


corpus = contract_fixtures.corpus
no_retrieval_or_model_loading = contract_fixtures.no_retrieval_or_model_loading


def test_query_modes_keep_source_order_and_structured_identity(
    corpus: PdfLibraryCorpus,
) -> None:
    golden = _golden(corpus)
    expected = {
        PdfQueryMode.ORIGINAL: [("both", None), ("which", None),
                                ("absent", None)],
        PdfQueryMode.CLARIFIED: [("which", "queues"), ("which", "producers")],
        PdfQueryMode.ALL: [
            ("both", None), ("which", None),
            ("which", "queues"), ("which", "producers"), ("absent", None),
        ],
    }
    before = golden.model_dump_json()
    for mode, keys in expected.items():
        plan = build_pdf_multibook_query_plan(golden, mode)
        assert [query.identity.key for query in plan.queries] == keys
        require_pdf_query_plan_matches(plan, golden)
        assert plan == PdfMultibookQueryPlan.model_validate_json(
            plan.model_dump_json(),
        )
    assert golden.model_dump_json() == before
    assert build_pdf_multibook_query_plan(golden).queries[1].track == (
        "original_ambiguous"
    )


@pytest.mark.parametrize("missing", [None, "", "  "])
def test_selected_clarified_question_cannot_be_invented(
    corpus: PdfLibraryCorpus, missing: str | None,
) -> None:
    data = _golden(corpus).model_dump(mode="json")
    data["cases"][1]["interpretations"][0]["clarified_question"] = missing
    golden = PdfMultibookGolden.model_validate_json(json.dumps(data))
    build_pdf_multibook_query_plan(golden, PdfQueryMode.ORIGINAL)
    for mode in (PdfQueryMode.CLARIFIED, PdfQueryMode.ALL):
        with pytest.raises(ValueError, match="Missing clarified"):
            build_pdf_multibook_query_plan(golden, mode)


@pytest.mark.parametrize("change", ["missing", "order", "text", "identity"])
def test_query_plan_must_match_exact_golden_variants(
    corpus: PdfLibraryCorpus, change: str,
) -> None:
    golden = _golden(corpus)
    data = build_pdf_multibook_query_plan(golden).model_dump(mode="json")
    if change == "missing":
        data["queries"].pop()
    elif change == "order":
        data["queries"].reverse()
    elif change == "text":
        data["queries"][0]["text"] = "Unapproved query text."
    else:
        data["queries"][2]["identity"]["interpretation_id"] = "unknown"
    plan = PdfMultibookQueryPlan.model_validate_json(json.dumps(data))
    with pytest.raises(ValueError, match="query plan differs"):
        require_pdf_query_plan_matches(plan, golden)


@pytest.mark.parametrize("change", ["duplicate", "track", "mode", "blank"])
def test_query_plan_rejects_inconsistent_metadata(
    corpus: PdfLibraryCorpus, change: str,
) -> None:
    data = _manifest(corpus).query_plan.model_dump(mode="json")
    if change == "duplicate":
        data["queries"].append(data["queries"][0])
    elif change == "track":
        data["queries"][0]["track"] = "clarified"
    elif change == "mode":
        data["mode"] = "original"
    else:
        data["queries"][0]["text"] = "  "
    with pytest.raises(ValidationError):
        PdfMultibookQueryPlan.model_validate_json(json.dumps(data))
