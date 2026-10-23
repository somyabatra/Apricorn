# Tests use the router's built-in node/edge DRC unless a test asks for the real shapely engine
# explicitly (test_e2e_real_drc). This keeps the suite deterministic and fast.
import os

os.environ.setdefault("APRICORN_DRC", "fallback")
