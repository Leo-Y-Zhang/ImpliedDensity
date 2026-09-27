# Changelog

Notable changes to this project, in the format of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The project has never been tagged for release, so there are no version headings
yet, only *Unreleased*. Back-filling release notes for work that shipped without
them would be writing history after the fact, which is the thing this
repository's documents are meant not to do.

## [Unreleased]

### Added

- **SVI smile fitting** with Gatheral and Jacquier's g(k) >= 0 butterfly check,
  making the recovered density verified non-negative rather than clipped into
  looking non-negative. On a live SPY chain this removed the negative mass
  entirely (16.0% -> 0.00%) and brought the mean to within +0.44% of the
  forward.
- **CLI** (`implieddensity density|chain|smile|verify`), packaging, and an
  offline correctness gate reachable as a first-class command.

### Fixed

- **Left-tail probabilities were understated.** The density is normalised on
  the quoted strike range (460-940 on the published chain), which silently
  dropped the 0.73% of risk-neutral mass below the lowest strike, and the tail
  integral stopped at the last grid point below the threshold rather than at
  the threshold. The mass outside the range is now read from the call slope
  and added back. Recomputed from the SVI fit recorded in `results.json` (the
  chain itself is not re-fetched): P(down 10%) 8.48% -> 9.21%, P(down 20%)
  2.21% -> 2.94%, lognormal P(down 10%) 9.37% -> 9.41% (the lognormal is now
  analytic), and the 20%-drawdown ratio to the lognormal 6.6x -> 8.6x. Skew,
  kurtosis and the figure are unchanged.

### Notes

- Smoothing the smile and truncating to liquid strikes were both tried before
  SVI. Both made the problem worse, and truncation additionally drove the
  20% -drawdown probability to zero, which is an artefact of cutting off the
  tail rather than a finding about it.

