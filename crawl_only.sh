# Crawler-only command for privacy-policy-annotator
# This runs ONLY metadata_crawl + policy_crawl (no LLM steps)

# 1. Install dependencies (if not already done)
source .venv/bin/activate

# 2. Run crawler only (skip all LLM steps)
python main.py \
  -run-id my_crawl_run \
  -pkg com.example.app \
  -model qwen3.6:35b \
  -hostname https://gateway.snet.tu-berlin.de/echelon/ollama \
  -skip-clean \
  -skip-detect \
  -skip-parse \
  -skip-annotate \
  -skip-review

# Output files will be at:
# Metadata: ../output/my_crawl_run/policies/metadata/APP_PACKAGE.jsonl
# HTML:     ../output/my_crawl_run/policies/html/APP_PACKAGE.html
