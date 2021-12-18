from multiprocessing import Lock, Process, Pipe, connection
import time, os, random

my_dict = {}
lock = Lock()


def opt(n):
    if n < 0:
        return 1, 1
    left = n // 2
    right = n - left
    return left, right


def dhs(n):
    print(os.getpid(), os.getppid(), time.process_time_ns(), "n: ", n)
    left, right = opt(n)
    print(
        os.getpid(),
        os.getppid(),
        time.process_time_ns(),
        "left:",
        left,
        "---- right:",
        right,
    )

    print(os.getpid(), os.getppid(), time.process_time_ns(), "my_dict:", my_dict)
    print(os.getpid(), os.getppid(), time.process_time_ns(), "my_dict id:", id(my_dict))
    print(
        os.getpid(),
        os.getppid(),
        time.process_time_ns(),
        "my_dict keys:",
        my_dict.keys(),
    )
    if n in my_dict.keys():
        print(
            os.getpid(), os.getppid(), time.process_time_ns(), "inside if\n", my_dict[n]
        )
        my_dict[n].append((left, right))
    else:
        print(os.getpid(), os.getppid(), time.process_time_ns(), "inside else")
        my_dict[n] = [(left, right)]
    print(os.getpid(), os.getppid(), time.process_time_ns(), "my_dict:", my_dict)

    if left == 1 or right == 1:
        return

    p2 = Process(target=dhs, args=(right,))
    # p1 = Process(target=dhs, args=(left,))
    # p1.start()
    print(os.getpid(), os.getppid(), time.process_time_ns(), "forking right")
    p2.start()
    print(os.getpid(), os.getppid(), time.process_time_ns(), "calling left")
    dhs(left)
    print(os.getpid(), os.getppid(), time.process_time_ns(), "left returned")

    # p1.join()
    p2.join()
    print(os.getpid(), os.getppid(), time.process_time_ns(), "right joined")
    return


if __name__ == "__main__":
    dhs(16)
    print(my_dict)
