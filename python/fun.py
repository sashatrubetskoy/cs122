import re

STATES = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
	'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia',
	'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
	'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
	'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri',
	'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey',
	'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
	'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
	'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
	'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']

NEW_YORK = ['New York state', 'New York State', 'upstate New York',
	'Upstate New York', 'state of New York', 'State of New York',
	'Empire State']

print('Reading scripts...')
f = open("scripts.csv")
raw = f.read()
print('Scripts read.')

# for state in STATES:
# 	occurrences = len(re.findall(r'in\s' + state, raw))
# 	print('in {}, {}'.format(state, occurrences))

occurrences = re.findall(r'.{20}Kansas.{20}', raw)
for o in occurrences[:100]:
	print(o)

