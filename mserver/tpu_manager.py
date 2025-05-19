import threading

class TpuManager:

    def __init__(self, device_count: int):
        self._device_count = device_count
        self._occupancy = [False] * device_count 
        self._lock = threading.Lock()

    def acquire_one_tpu(self):
        self._lock.acquire()
        for i in range(self._device_count):
            if not self._occupancy[i]:
                self._occupancy[i] = True
                self._lock.release()
                return i
        self._lock.release()
        return -1

    def release_one_tpu(self, index: int):
        self._lock.acquire()
        self._occupancy[index] = False
        self._lock.release()
