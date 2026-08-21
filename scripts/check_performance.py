#!/usr/bin/env python3
"""Check performance budgets against benchmark results."""

import json
import sys

def check_performance_budgets(benchmark_file: str) -> bool:
    """Check if benchmark results meet performance budgets."""
    with open(benchmark_file) as f:
        data = json.load(f)

    budgets = {
        'basket_penetration': 5000,
        'basket_penetration_over_time': 500,
        'basket_composition': 500,
        'customer_entropy': 1000,
        'cohort_sizes': 500,
        'cohort_retention': 1000,
        'assortment': 5000,
        'rules_fpgrowth': 2000,
        'clv_bg_nbd': 60000,
        'clv_gamma_gamma': 60000,
        'performance_abc': 500,
    }

    failed = []
    for module in data['modules']:
        name = module['name']
        duration = module.get('duration_ms', 0)
        budget = budgets.get(name, 5000)
        if duration > budget:
            failed.append(f'{name}: {duration:.0f}ms > {budget}ms')

    if failed:
        print('PERFORMANCE REGRESSION DETECTED:')
        for f in failed:
            print(f'  - {f}')
        return False
    else:
        print('All performance budgets met!')
        return True


def check_performance_regression(main_file: str, pr_file: str) -> bool:
    """Check if PR branch has performance regressions compared to main."""
    with open(main_file) as f:
        main = json.load(f)
    with open(pr_file) as f:
        pr = json.load(f)

    main_lookup = {m['name']: m for m in main['modules']}
    pr_lookup = {m['name']: m for m in pr['modules']}

    regressions = []
    for name, main_m in main_lookup.items():
        pr_m = pr_lookup.get(name)
        if not pr_m:
            continue
        main_dur = main_m.get('duration_ms', 0)
        pr_dur = pr_m.get('duration_ms', 0)
        if main_dur > 0 and pr_dur > main_dur * 1.5:
            regressions.append(f'{name}: {pr_dur:.0f}ms vs {main_dur:.0f}ms ({pr_dur/main_dur:.1f}x)')

    if regressions:
        print('PERFORMANCE REGRESSION DETECTED:')
        for r in regressions:
            print(f'  - {r}')
        return False
    else:
        print('No significant performance regressions!')
        return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-budgets', help='Benchmark file to check budgets')
    parser.add_argument('--check-regression', nargs=2, metavar=('MAIN', 'PR'), help='Main and PR benchmark files for regression check')
    args = parser.parse_args()

    if args.check_budgets:
        success = check_performance_budgets(args.check_budgets)
        sys.exit(0 if success else 1)
    elif args.check_regression:
        main_file, pr_file = args.check_regression
        success = check_performance_regression(main_file, pr_file)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)