


import multiprocessing
import os
import time

#processing queue
input_queue = multiprocessing.Queue(maxsize=5)
output_queue = multiprocessing.Queue(maxsize=5)

def worker_main(in_queue, out_queue):
    #print os.getpid(),"working"
    while True:
        item = in_queue.get(True)
        #print os.getpid(), "got", item
        time.sleep(2) # simulate a "long" operation
        out_queue.put(item + "-Done")

the_pool = multiprocessing.Pool(5, worker_main,(input_queue, output_queue,))
#                            don't forget the coma here  ^

for q in range(10):
    print "Adding minibatch to processing queue: %s" % q
    for i in range(5):
        print("%s - Put HEllo" % i)
        input_queue.put("hello")
        
    print("\tIN:%s OUT:%s") % (input_queue.full(), output_queue.full())



time.sleep(10)