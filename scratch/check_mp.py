from multiprocessing import Process, Pipe, connection
import time
import os


def opt(n, conn):
    conn.send(n // 2)
    print(n, "just sent")
    print("now sleep for n sec")
    print("parent process:", os.getppid())
    print("process id:", os.getpid())
    time.sleep(n)
    conn.close()


def opt2(n):
    print("process id:", os.getpid(), "Start sleeping")
    time.sleep(n)
    print("process id:", os.getpid(), "End sleeping")


def pr(n):
    parent_conn, child_conn = Pipe()
    p = Process(target=opt, args=(n, child_conn))
    p.start()
    a = parent_conn.recv()
    print(a)
    if a != 1:
        pr(a)
    # p.join()


def pr2(n):
    # for i in range(n):
    p = Process(target=opt2, args=(5,))
    p2 = Process(target=opt2, args=(2,))
    p.start()
    p2.start()
    # p.join()
    # p2.join()


if __name__ == "__main__":
    pr(10)
    pr2(3)
