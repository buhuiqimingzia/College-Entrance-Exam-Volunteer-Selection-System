from multiprocessing import Process, Semaphore

def worker(semaphore, data):
    semaphore.acquire()
    try:
        # 访问共享资源的代码
        print("Worker:", data)
    finally:
        semaphore.release()

if __name__ == '__main__':
    semaphore = Semaphore(2)  # 设置允许的进程数
    processes = []
    for i in range(5):
        p = Process(target=worker, args=(semaphore, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
