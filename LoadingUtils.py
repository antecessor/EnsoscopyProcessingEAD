import numpy as np
import cv2


def read_img(imfile):
    return cv2.imread(imfile)[:, :, ::-1]


def read_boxes(txtfile):
    lines = []

    with open(txtfile, "r") as f:
        for line in f:
            line = line.strip()
            box = np.hstack(line.split()).astype(np.float)
            box[0] = int(box[0])
            lines.append(box)

    return np.array(lines)


def yolo2voc(boxes, imshape):
    m, n = imshape[:2]

    box_list = []
    for b in boxes:
        cls, x, y, w, h = b

        x1 = (x - w / 2.)
        x2 = x1 + w
        y1 = (y - h / 2.)
        y2 = y1 + h

        # absolute:
        x1 = x1 * n;
        x2 = x2 * n
        y1 = y1 * m;
        y2 = y2 * m

        box_list.append([cls, x1, y1, x2, y2])

    if len(box_list) > 0:
        box_list = np.vstack(box_list)

    return box_list


def plot_boxes(ax, boxes, labels):
    import seaborn as sns

    color_pal = sns.color_palette('hls', n_colors=len(labels))

    for b in boxes:
        cls, x1, y1, x2, y2 = b
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], lw=2, color=color_pal[int(cls)])

    return []


def read_obj_names(textfile):
    import numpy as np
    classnames = []

    with open(textfile) as f:
        for line in f:
            line = line.strip('\n')
            if len(line) > 0:
                classnames.append(line)

    return np.hstack(classnames)


def visualize(**images):
    """PLot images in one row."""
    import matplotlib.pyplot as plt
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


def readImageAndLabel(imgfile, bboxfile, show=False):
    """
    Example script to read and plot bounding box annotations (which are provided in <x,y,w,h> format)

    (x,y) - box centroid
    (w,h) - width and height of box in normalised. 
    :rtype: object
    """

    img = read_img(imgfile)
    boxes = read_boxes(bboxfile)

    # convert boxes from (x,y,w,h) to (x1,y1,x2,y2) format for plotting
    boxes_abs = yolo2voc(boxes, img.shape)

    if show:
        showImageAndLabels(boxes_abs, [1, 2, 3, 4, 5, 6], img)
    return img, boxes_abs


def showImageAndLabels(boxes_abs, classes, img):
    import pylab as plt
    fig, ax = plt.subplots()
    ax.imshow(img)
    plot_boxes(ax, boxes_abs, classes)
    plt.show()
