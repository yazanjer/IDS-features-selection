#!/usr/bin/env bash
# ----------------------------------------------------------------------
# push_to_github.sh
#
# One-shot script to publish ids-bgwo-shap/ to:
#   https://github.com/yazanjer/IDS-features-selection
#
# Usage:
#   cd "/Users/yazanaljeroudi/Documents/feature selection IDS/ids-bgwo-shap"
#   chmod +x push_to_github.sh
#   ./push_to_github.sh
#
# When `git push` runs, it will prompt for credentials:
#   Username: yazanjer
#   Password: <paste your GitHub Personal Access Token here>
#            (not your account password — GitHub disabled that years ago)
#
# Generate a PAT at: https://github.com/settings/tokens
#   - Classic token: tick `public_repo` scope.
#   - Fine-grained token: scope it to this repo, give it
#       Contents: Read and Write + Metadata: Read.
#   Set a short expiry (7 days is fine — revoke after the push lands).
#
# This script is idempotent: safe to re-run if the first push failed
# (e.g. wrong PAT). It will skip re-init / re-commit if already done.
# ----------------------------------------------------------------------
set -euo pipefail

REPO_URL="https://github.com/yazanjer/IDS-features-selection.git"
USER_NAME="Yazan Aljeroudi"
USER_EMAIL="yazan.aljeroudi@gmail.com"
BRANCH="main"
COMMIT_MSG="Initial commit: IDS-BGWO-SHAP — BGWO feature selection + SHAP-in-the-loop extension of LCCDE

Two contributions on top of the LCCDE intrusion-detection ensemble
(Yang et al., GLOBECOM '22):

  1. Binary Grey Wolf Optimizer (from scratch) replacing the baseline's
     information-gain/FCBF feature selection — src/bgwo_fs.py.

  2. SHAP-in-the-loop explanation-coherence term injected into the BGWO
     fitness function, making it tri-objective (accuracy + sparsity +
     SHAP coherence) — src/fitness.py. Setting gamma=0 recovers the
     bi-objective ablation isolating the SHAP contribution.

Downstream LCCDE classifier held fixed across all four FS branches
(none / filter / bgwo_bi / bgwo_shap) so any delta is attributable to
the feature-selection stage alone.

Includes:
  - faithful vectorised LCCDE reimplementation (src/lccde_model.py)
  - Kaggle + local-CSV data loaders for CIC-IDS2017 and UNSW-NB15
  - multi-seed runner with Wilcoxon signed-rank tests vs reference
  - per-attack-class SHAP signatures, LIME baseline, Kuncheva stability,
    explanation fidelity
  - all plots: per-class F1, confusion, SHAP summary/per-class,
    BGWO convergence, |S|-vs-F1 Pareto, cross-dataset overlap,
    latency-vs-features, comparison table
  - thin Colab launcher with interactive kaggle.json upload (notebooks/)
  - smoke test on tiny config (tests/smoke_test.py) — passing
  - baseline LCCDE notebook preserved unmodified under baseline/

Citations to the four Yang/Shami papers in README."

# ----------------------------------------------------------------------
# 0. Sanity check — must be run from the project root.
# ----------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "README.md" || ! -d "src" ]]; then
    echo "ERROR: this script must live next to README.md / src/."
    echo "       cwd is now: $SCRIPT_DIR"
    exit 1
fi

echo "==> Working in: $SCRIPT_DIR"

# ----------------------------------------------------------------------
# 1. Initialise git if not already a repo.
# ----------------------------------------------------------------------
if [[ ! -d .git ]]; then
    echo "==> git init -b $BRANCH"
    git init -b "$BRANCH"
else
    echo "==> .git already exists, skipping init"
    # Make sure we're on main.
    current_branch=$(git symbolic-ref --quiet --short HEAD 2>/dev/null || echo "")
    if [[ "$current_branch" != "$BRANCH" ]]; then
        git checkout -B "$BRANCH"
    fi
fi

# ----------------------------------------------------------------------
# 2. Identity (local to this repo only — doesn't touch your global config).
# ----------------------------------------------------------------------
git config user.name  "$USER_NAME"
git config user.email "$USER_EMAIL"
echo "==> Author: $(git config user.name) <$(git config user.email)>"

# ----------------------------------------------------------------------
# 3. Stage everything that .gitignore allows.
# ----------------------------------------------------------------------
echo "==> Staging files..."
git add .

# ----------------------------------------------------------------------
# 4. Belt-and-suspenders: refuse to commit if anything sensitive snuck in.
# ----------------------------------------------------------------------
echo "==> Sensitive-file check..."
if git diff --cached --name-only | grep -iE 'kaggle\.json|\.venv|__pycache__|\.env$|_token|secret' >/dev/null; then
    echo "ERROR: refusing to commit — sensitive file detected in the index:"
    git diff --cached --name-only | grep -iE 'kaggle\.json|\.venv|__pycache__|\.env$|_token|secret'
    echo "Fix .gitignore or run 'git rm --cached <file>' and re-run this script."
    exit 1
fi
echo "    OK — no sensitive files staged."

# ----------------------------------------------------------------------
# 5. Commit (skip if there's nothing to commit, e.g. on re-runs).
# ----------------------------------------------------------------------
if git diff --cached --quiet; then
    echo "==> No staged changes — skipping commit."
else
    echo "==> git commit"
    git commit -m "$COMMIT_MSG"
fi

# Make sure at least one commit exists before pushing.
if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
    echo "ERROR: no commits to push."
    exit 1
fi

# ----------------------------------------------------------------------
# 6. Configure the remote.
# ----------------------------------------------------------------------
if git remote get-url origin >/dev/null 2>&1; then
    existing=$(git remote get-url origin)
    if [[ "$existing" != "$REPO_URL" ]]; then
        echo "==> Updating origin: $existing -> $REPO_URL"
        git remote set-url origin "$REPO_URL"
    else
        echo "==> origin already set to $REPO_URL"
    fi
else
    echo "==> Adding origin -> $REPO_URL"
    git remote add origin "$REPO_URL"
fi

# ----------------------------------------------------------------------
# 7. Push.
#    Git will prompt for username + password (PAT) here.
#    macOS Keychain may also offer to store the PAT — say no if you
#    plan to revoke it after this push.
# ----------------------------------------------------------------------
echo
echo "==> Pushing to $REPO_URL on branch $BRANCH"
echo "    When prompted:"
echo "      Username: yazanjer"
git push -u origin "$BRANCH"

# ----------------------------------------------------------------------
# 8. Verify.
# ----------------------------------------------------------------------
echo
echo "==> Verifying remote..."
git ls-remote origin "$BRANCH" || true

echo
echo "==> Done."
echo "    Repo: https://github.com/yazanjer/IDS-features-selection"
echo "    Don't forget to revoke the PAT now if it was a one-shot token:"
echo "    https://github.com/settings/tokens"
