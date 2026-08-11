# Data Scientist Sub-Agent Task

## Prompt
Review the new switching analytics functions for statistical correctness:
1. compute_substitution_strength - validate the statistical approach for classifying switching patterns
2. compute_high_value_switching - review the methodology for identifying valuable customer switching
3. generate_switching_opportunity_matrix - assess the decision framework for prioritizing actions
4. Check that all functions properly handle edge cases and return meaningful values
5. Verify that the functions align with the documented intent in their docstrings

## Context
This task relates to the recent changes in the market_basket_app repository where:
- The promo module was refactored (src/analytics/promo.py deleted, functionality moved to src/analytics/promo/ directory)
- Significant enhancements were made to switching analytics in src/analytics/switching.py
- New functions were added: compute_switching_status, compute_switch_in_out_rates, compute_substitution_strength, compute_high_value_switching, generate_switching_opportunity_matrix
- A comprehensive test file was created at tests/unit/test_switching.py to cover these new functions

## Files to Review
- src/analytics/switching.py (particularly the new functions)
- src/analytics/schemas.py (to understand the data contracts)
- tests/unit/test_switching.py (the test file I created)
- Any usage of these new functions throughout the codebase