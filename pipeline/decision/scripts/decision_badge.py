import json
import anybadge

with open('decision.json') as f:
    decision = json.load(f)
    passed = decision['passed']
    badge = None
    if passed:
        badge = anybadge.Badge('Audit', 'Passed ✅', default_color='green')
    else:
        badge = anybadge.Badge('Audit', 'Failed ❌', default_color='darkred')
    badge.write_badge('decision_badge.svg', overwrite=True)
