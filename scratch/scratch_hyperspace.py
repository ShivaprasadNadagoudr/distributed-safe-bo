ss = [[1, 2, 3, 4], [5, 6, 7, 8]]
rows = len(ss)
columns = len(ss[0])
print(rows, columns)

items = []
for i in range(columns**rows):
    items.append([])

for row in range(rows):
    repeat = columns ** (rows-row-1)
    # print("repeat", repeat)
    for column in range(columns):
        item = ss[row][column]
        # print("item", item)
        start = column * repeat
        for times in range(columns**row):
            # print("start", start)
            for l in range(repeat):
                items[start+l].append(item)
                # print(start+l)
            start += columns * repeat
print(items)
