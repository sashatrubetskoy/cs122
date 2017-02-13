f = open("data/scripts.csv", 'rb')
raw = f.readlines()
f.close()

f = open("data/scripts.csv", 'wt')

header = ['"title", "script"']
raw = header + raw

print('Writing lines...')
for line in raw:
	f.write(line)

print('done')
f.close()

