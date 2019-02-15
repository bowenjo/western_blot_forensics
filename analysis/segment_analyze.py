import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm


class AnalyzeIouRecall(object):
    def __init__(self, RPN, iou_list):
        """
        RPN: Region Proposal Network object
        """
        self.RPN = RPN
        self.iou_list = iou_list
        self.p_truth = {}
        self.r_truth = {}

        for iou in iou_list:
            self.p_truth[iou] = []
            self.r_truth[iou] = []

    def run(self, data):
        i = 0
        for gt_boxes, image in zip(data["rects"], data["image"]):
            f, l = self.RPN.blotExtractor(image.copy())
            if l is not None:
                wb_boxes = cvtRect(l["loc"])
            else:
                wb_boxes = np.array([]).reshape(0,4) 
            self.analyze(gt_boxes, wb_boxes)
            i += 1

            if i% 100 == 0:
                print("Image {} of {} completed".format(i, len(data["image"])))

        recall = [calc_recall(self.r_truth[key]) for key in np.sort(list(self.r_truth.keys()))]
        # precision = [calc_precision(self.p_truth[key]) for key in np.sort(list(self.p_truth.keys()))]

        return recall#, precision

    def analyze(self, gt_boxes, pred_boxes):
        for iou in self.iou_list:
            self.iou_recall(pred_boxes, gt_boxes, iou)

    def iou_recall(self, predictions, gt_boxes, iou):
        iou_gt_predictions = np_iou(gt_boxes, predictions)
        truth_bb = iou_gt_predictions >= iou

        p_truth_array = np.all(1 - truth_bb, axis=0)
        r_truth_array = np.any(truth_bb, axis=1)

        self.p_truth[iou] += list(p_truth_array)
        self.r_truth[iou] += list(r_truth_array)


def calc_recall(r_truth):
    return np.sum(r_truth) / len(r_truth)

def calc_precision(r_truth, p_truth):
    tp = np.sum(r_truth)
    fp = np.sum(p_truth)
    return tp / (tp+fp)

def cvtRect(rect):
    if rect is not None and len(rect) > 0:  
        rect_out = rect.copy()
        rect_out[:, 2:] = rect[:,:2] + rect[:, 2:]
    else:
        rect_out = np.array([]).reshape(0,4)
    
    return rect_out.astype(np.float32)

def np_iou(boxes1, boxes2):
    """Computes pairwise intersection-over-union between box collections.
    Args:
    boxes1: a numpy array with shape [N, 4] holding N boxes.
    boxes2: a numpy array with shape [M, 4] holding M boxes.
    Returns:
    a numpy array with shape [N, M] representing pairwise iou scores.
    """
    intersect = np_intersection(boxes1, boxes2)
    area1 = np_area(boxes1)
    area2 = np_area(boxes2)
    union = np.expand_dims(area1, axis=1) + np.expand_dims(area2, axis=0) - intersect
    return intersect / union

def np_area(boxes):
    """Computes area of boxes.
    Args:
    boxes: Numpy array with shape [N, 4] holding N boxes
    Returns:
    a numpy array with shape [N*1] representing box areas
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def np_intersection(boxes1, boxes2):
    """Compute pairwise intersection areas between boxes.
    Args:
    boxes1: a numpy array with shape [N, 4] holding N boxes
    boxes2: a numpy array with shape [M, 4] holding M boxes
    Returns:
    a numpy array with shape [N*M] representing pairwise intersection area
    """
    [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
    [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    intersect_heights = np.maximum(np.zeros(all_pairs_max_ymin.shape, dtype='f4'), all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    intersect_widths = np.maximum(np.zeros(all_pairs_max_xmin.shape, dtype='f4'), all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths

def plot_iou_recall(iou, recalls, names, metric):
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    for name, recall in zip(names,recalls):
        ax.plot(iou, recall, label = name, linewidth=3, marker = 's', markersize = 7,)

    cmap_gray = cm.get_cmap("gray")    
    ax.set_xlabel("IOU")
    ax.set_ylabel(metric)
    ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.grid(which="major", color=cmap_gray(.8), linestyle='--', linewidth=1)
    ax.set_axisbelow(True)

    ax.legend()
    plt.show()







