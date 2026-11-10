# Checkout agent (TouchStone demo)

A tiny RL environment for an LLM checkout agent, used by the TouchStone interactive demo.
The agent (`agent/`) places an order; the grader (`grader/grader.py`) scores it.

The grader has a planted bug: it rewards an order shipped to *any* address, although the task
rules (`touchstone/RULES.md`) say to ship to the address the user gave. The TouchStone suite in
`touchstone/` holds three approved tests, all passing, so the bug goes unnoticed.

In the demo, each visitor gets their own organization and branch (`demo/...`):

1. **Propose tests.** In demo organizations the proposals are predetermined
   (`touchstone/demo.json`); no LLM is called. Everything else runs for real.
2. **Check against grader** runs the proposals on GitHub Actions: "Wrong address scores 0"
   disagrees with the grader, a likely bug.
3. **Approve** and **Save to suite** commit the tests and their content-bound approvals to the
   branch through a pull request.
4. **Run approved tests**: the new test fails.
5. **Fix** applies the one-line grader fix (`demo.json`) through a pull request; run again and
   every test passes.

Run it locally: `pip install touchstone-us` then `touchstone check touchstone`.
