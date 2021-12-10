no_subspaces = 8
no_params = 3
search_space = []
deployment_status = []

for i in range(no_params):
    search_space.append([0, no_subspaces - 1])
# print(search_space)

for i in range(no_params):
    parameter_deployment_status = []
    for j in range(no_subspaces):
        parameter_deployment_status.append(False)
    deployment_status.append(parameter_deployment_status)
# print(deployment_status)

hyperspaces = [[6, 2, 3], [6, 6, 4], [5, 2, 3], [4, 5, 6]]


def split_space(ss, hs1, hs2):
    print()
    print(ss)
    print(hs1, hs2)
    ss1 = []
    ss2 = []
    for param_index, (i, j) in enumerate(zip(hs1, hs2)):
        print(param_index)
        print(i, j)
        print(ss[param_index])
        start, end = ss[param_index]
        if i == j:
            # print("in IF")
            ss1.append(ss[param_index])
            ss2.append(ss[param_index])
        else:
            # print("in ELSE")
            if i > j:
                i, j = j, i
            ds = deployment_status[param_index]
            ds[i] = True
            ds[j] = True
            in_between = j - i - 1
            hs1_right = in_between // 2
            hs2_left = in_between - hs1_right
            hs1_low = i
            hs1_high = i + hs1_right
            hs2_low = j - hs2_left
            hs2_high = j

            for k in range(i - 1, start - 1, -1):
                if not ds[k]:
                    hs1_low = k
                else:
                    break

            for k in range(j + 1, end + 1):
                if not ds[k]:
                    hs2_high = k
                else:
                    break

            ss1.append([hs1_low, hs1_high])
            ss2.append([hs2_low, hs2_high])
            break
    if param_index < no_params - 1:
        for i in range(param_index + 1, no_params):
            ss1.append(ss[i])
            ss2.append(ss[i])
    return ss1, ss2


ss1, ss2 = split_space(search_space, hyperspaces[0], hyperspaces[1])
print(ss1)
print(ss2)
# print(deployment_status)
print(split_space(ss1, hyperspaces[0], hyperspaces[2]))
print(split_space(ss2, hyperspaces[1], hyperspaces[3]))

# print(deployment_status)
