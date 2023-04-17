import pandas as pd

# read raw tables
tables = {}
defaults = {}
for i in range(1,4):
    tables[i] = pd.read_csv("excels/" + str(i) + "-firms-mc-raw.csv")
    defaults[i] = []
    for j in range(100):
        defaults[i].append(tables[i].loc[0, 'time_of_default_' + str(j)])

for t in range(1,16):
    alive = [sum(map(lambda x : x>=t, defaults[i])) for i in range(1,4)]
    print("Period " + str(t) + ": " + str(alive))
