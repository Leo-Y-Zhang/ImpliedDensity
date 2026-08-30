# Contributing to ImpliedDensity

## What this project is

It recovers the risk-neutral probability distribution implied by a
live option chain, and checks the no-arbitrage condition of the fit
rather than assuming it.

It is published so that its claims can be checked, not as a component to build
on. The licence grants the right to read it, run it, and publish what you find
-- including a refutation -- and nothing beyond that.

## The bar for a change

**A test that has never been observed failing is decoration.** If you add one,
break the code on purpose first and confirm the test notices. Several bugs in
this repository were caught exactly that way, and at least one was a bug in the
checking harness rather than in the code it was checking.

**Numbers in the README are generated, never typed.** They are injected from
the result files by `make_readme.py`, which fails if a value is missing. If you
change an analysis, regenerate rather than editing the prose to match.

**Report what the data says.** If a control moves a result the wrong way, that
is the finding. Do not quietly pick the specification that flatters the
conclusion.

## Before opening anything

```bash
python -m unittest discover -s tests -v
python scripts/check_spdx.py
ruff check .
```

## DCO, not a CLA

Contributions are accepted under the [Developer Certificate of
Origin](https://developercertificate.org/). Sign off each commit:

```bash
git commit -s -m "your message"
```

A contributor licence agreement would need a named legal entity to assign
rights to. This project is maintained under a pseudonymous account, and creating
that entity would defeat the point, so the DCO does the necessary job instead:
you assert you have the right to contribute what you are contributing.

## Security

Please report vulnerabilities privately -- see [SECURITY.md](SECURITY.md).
