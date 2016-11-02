
import multiprocessing
from dataset_preparation.kitti_generator import KittiGenerator
import time
import traceback
import cv2

def worker_extraction(params, out_queue):
    cv2.setNumThreads(-1)
    generator = KittiGenerator(params.kitti_path,
                               params.base_depth,
                               params.depth_step)
    while True:
        try:
            batch = generator.next_batch(num_set_same_img=params.patches_per_set)
            #print("Done extraction, put in queue.")
            out_queue.put(batch)
        except Exception as e:
            print("Extraction Worker Exception: %s" % e)
            traceback.print_exc()


def worker_aggregation(psw_queue, ready_queue, input_organizer, aggregation_size):
    cv2.setNumThreads(-1)
    while True:
        try:
            planes_list = []
            #print("Aggregating %s PSW" % aggregation_size)
            for i in xrange(aggregation_size):
                #print("Getting plane for aggregation %s" % i)
                planes_list.append(psw_queue.get())

            # Aggregate queue
            start_time = time.time()
            feed_dict = input_organizer.get_feed_dict(planes_list)
            duration = time.time() - start_time

            # Put aggregation in batch
            print("Done aggregation! (duration: %.3f sec)" % duration)
            ready_queue.put(feed_dict)
        except Exception as e:
            print("Aggregation Worker Exception: %s" % e)
            traceback.print_exc()


class KittiParams(object):
    def __init__(self, kitti_path, base_depth, depth_step, patches_per_set):
        self.kitti_path = kitti_path
        self.base_depth = base_depth
        self.depth_step = depth_step
        self.patches_per_set = patches_per_set


class GeneratorQueued(object):

    def __init__(self, kitti_params, input_organizer, batch_size=5, extraction_workers=5, aggregation_workers=1):
        # Params
        self.batch_size = batch_size
        self.aggregation_workers = aggregation_workers
        self.extraction_workers = extraction_workers
        self.aggregation_size = self.batch_size / kitti_params.patches_per_set

        print("Generator Queue: (%s - plane extractors) (%s - plane aggregators)" % (self.extraction_workers,
                                                                                     self.aggregation_workers))
        print("Aggregator size: (%s) for batch_size=%s" % (self.aggregation_size, batch_size))

        # kitti dataset parameters
        self.kitti_params = kitti_params

        self.input_organizer = input_organizer

        # Queue that has output from extracted PSW
        self.psw_queue = multiprocessing.Queue(maxsize=self.extraction_workers)

        # Queue that has aggregations in batches of PSW
        self.ready_queue = multiprocessing.Queue(maxsize=self.aggregation_workers)

        cv2.setNumThreads(0)

        # Extract PSW
        self.extraction_pool = multiprocessing.Pool(self.extraction_workers, worker_extraction, (self.kitti_params, self.psw_queue,))

        # Aggregate PSW in batches
        self.batch_pool = multiprocessing.Pool(self.aggregation_workers, worker_aggregation, (self.psw_queue, self.ready_queue, self.input_organizer, self.aggregation_size,))

    def get_batch(self):
        return self.ready_queue.get()


