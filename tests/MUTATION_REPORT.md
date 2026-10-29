# Mutation-guided unit tests

One unittest file, `tests/test_chain_and_edges.py` (9 tests), and a reproducible
mutation runner, `tests/mutation_check.py`, with its results in
`tests/mutation_results.json`. Production code and CI are unchanged: CI already
runs `python -m unittest discover -s tests`.

## Measurement

`python tests/mutation_check.py before` runs the CI gate (SPDX check, ruff,
unittest) against each mutant with only the original `test_density.py`;
`after` runs it with every test file. Each mutant is applied alone in a
temporary copy; the original sources are never edited.

| Module | Mutants tried | Killed before | Killed after |
| --- | ---: | ---: | ---: |
| `src/implieddensity/chain.py` | 12 | 2 | 12 |
| `src/implieddensity/blackscholes.py` | 7 | 5 | 7 |
| `src/implieddensity/density.py` | 3 | 2 | 3 |
| `src/implieddensity/svi.py` | 3 | 1 | 3 |
| **Total** | **25** | **10** | **25** |

The round-trip tests price a clean chain and recover it, so they never saw what
the chain parser discards: puts, other expiries, one-sided, crossed and
below-minimum quotes all survived. This is a targeted sample, not an exhaustive
mutation score.

## What each test kills

| Test | Mutant killed (line) |
| --- | --- |
| `test_keeps_only_usable_calls_of_the_expiry_sorted_by_strike` | `chain.py:82` puts or other expiries kept; `:87` crossed quotes kept, minimum bid made exclusive; `:90` mid replaced by the ask; `:93` spread negated; `:95` strikes left unsorted |
| `test_one_sided_quotes_are_dropped_even_without_a_minimum_bid` | `chain.py:85` one-sided quotes kept |
| `test_expiries_are_distinct_and_ascending_and_skip_unparseable_rows` | `chain.py:69` expiries in descending order |
| `test_refresh_bypasses_the_cache_and_rewrites_it` | `chain.py:49` `use_cache=False` still reads the cache |
| `test_zero_vol_exactly_at_the_forward_is_worth_nothing_not_nan` | `blackscholes.py:43` zero vol not treated as degenerate (gives NaN at the money) |
| `test_vega_is_the_derivative_of_the_price_in_vol` | `blackscholes.py:61` vega missing its sqrt(tau) |
| `test_a_smile_with_butterfly_arbitrage_is_flagged` | `density.py:128` arbitrage flag always true; `svi.py:75` g(k) missing the 1/2 on w'' |
| `test_fit_needs_six_points` | `svi.py:43` five points accepted |

The arbitrage test uses Axel Vogt's SVI parameters, the standard example of a
smile with butterfly arbitrage: the pipeline recovers them (vol RMSE about 2e-9)
and must report g < 0 and `butterfly_arbitrage_free = False`.

## Commands and results

Python 3.13 on a real checkout:

| Command | Result |
| --- | --- |
| `python tests/mutation_check.py before` | 10/25 killed |
| `python tests/mutation_check.py after` | 25/25 killed |
| `python -m unittest discover -s tests` | 51 tests, OK |
| `ruff check .` | All checks passed |
| `python scripts/check_spdx.py` | all source files carry the identifier |

No test makes a network request: the one fetch test replaces
`urllib.request.urlopen` and uses a temporary cache directory.
