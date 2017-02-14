import matplotlib.pyploy as plt

file1 = sys.argv[1]
file2 = sys.argv[2]
file3 = sys.argv[3]
file4 = sys.argv[4]

x1 = []
x2 = []
x3 = []
x4 = []
y1 = []
y2 = []
y3 = []
y4 = []

with open(file1, 'r') as f:
    lines = f.readlines()
    for l in lines:
        tokens = l.split()
        x1.append(tokens[0])
        y1.append(tokens[1])

with open(file2, 'r') as f:
    lines = f.readlines()
    for l in lines:
        tokens = l.split()
        x2.append(tokens[0])
        y2.append(tokens[1])

with open(file3, 'r') as f:
    lines = f.readlines()
    for l in lines:
        tokens = l.split()
        x3.append(tokens[0])
        y3.append(tokens[1])

with open(file4, 'r') as f:
    lines = f.readlines()
    for l in lines:
        tokens = l.split()
        x4.append(tokens[0])
        y4.append(tokens[1])

plt.plot(x1, y1)
plt.plot(x2, y2)
plt.plot(x3, y3)
plt.plot(x4, y4)

plt.legend(['head-pred-unfiltered', 'head-pred-filtered', 'tail-pred-unfiltered', 'tail-pred-filtered'], loc='upper left')
