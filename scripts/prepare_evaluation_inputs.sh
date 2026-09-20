#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

mkdir -p \
    "$project_root/data/raw" \
    "$project_root/data/datasets/AnsweredQuestions" \
    "$project_root/data/datasets/UnansweredQuestions"

cat <<EOF
Evaluation input directories are ready:

  Corpus:
    $project_root/data/raw/

  Ground-truth datasets:
    $project_root/data/datasets/AnsweredQuestions/

  Question datasets used for search:
    $project_root/data/datasets/UnansweredQuestions/

Extract or copy the evaluator-provided files into these directories.
The project creates data/processed/ and data/output/ when commands run.
EOF
