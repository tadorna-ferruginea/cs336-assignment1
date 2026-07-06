# manually create max heap
def max_heap_move_down(heap: list, index: int):
    while True:
        child0 = (index + 1) * 2 - 1
        child1 = child0 + 1
        if child0 >= len(heap):
            return
        else:
            big_child = child0
            if child1 < len(heap):
                if heap[child1] > heap[child0]:
                    big_child = child1

            # swap or stop
            if heap[index] >= heap[big_child]:
                return
            else:
                heap[index], heap[big_child] = heap[big_child], heap[index]

                # update index
                index = big_child


def max_heap_move_up(heap: list, index: int):
    while True:
        mom = (index + 1) // 2 - 1
        if heap[mom] < heap[index]:
            heap[mom], heap[index] = heap[index], heap[mom]
            index = mom
            if index == 0:
                return
        else:
            return


def max_heapify(pre_heap: list):
    for i in range(len(pre_heap) // 2 - 1, -1, -1):
        max_heap_move_down(pre_heap, i)


def max_heap_pop(heap: list):
    result = heap[0]
    last = heap.pop()
    if heap:
        heap[0] = last
        max_heap_move_down(heap, 0)
    return result


def max_heap_push(heap: list, element):
    heap.append(element)
    max_heap_move_up(heap, len(heap) - 1)
