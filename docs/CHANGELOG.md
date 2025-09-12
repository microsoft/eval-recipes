# Changelog

Use this file to assist with upgrading to new versions.


## [0.0.6]

**Improved claim verification evaluation by introducing tunable "probabilities"**

### Added
- Added `verified_probability` and `open_domain_probability` as configurable thresholds in `ClaimVerificationEvaluatorConfig`. This allows for changing how sensitive the evaluation is to marking a claim as not supported or open domain.

### Changed
- Fixed typos.
- **BREAKING**: Changed claim verification metadata output field names:
  - `number_supported_claims` -> `num_closed_domain_supported` 
  - `number_not_supported_claims` -> removed (can be calculated as `total_claims_closed_domain - num_closed_domain_supported`)
  - `number_open_domain_claims` -> `num_open_domain_claims`
  - Added `total_claims_closed_domain` field to track closed domain claims separately
- Simplified the script `tests/validate_evaluations.py`


## [0.0.5]

**Initial release**
