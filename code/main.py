import cv2
import numpy as np
import os


def show_wrap(name, img, size, close=True):
    """
    :param name: [string] - Name of the window to be created
    :param img: [np.ndarray] - Image to be shown
    :param size: [int] - Resizing of the window
    :param close: [bool] - If 'False' window stays on screen (for comparisons)
    :return: null

    Simple wrapper for cv2.imshow(). Resize with original aspect ratio, display, wait for keystroke and close.
    """
    h, w = img.shape[:2]
    aspect_ratio = min(w, h) / max(w, h)
    img_res = cv2.resize(img, (size, np.int0(size * aspect_ratio)))
    cv2.imshow(name, img_res)
    cv2.waitKey(0)
    if close:
        cv2.destroyAllWindows()


def load_images_from_folder(folder):
    """
    :param folder: [string] - path to folder with images
    :return: [np.array of np.ndarray, np.array] - array of images and array of corresponding filenames

    Load images from folder
    """
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames


def empty_beach_mask(gray, show_detailed_steps=False):
    """
    :param gray: [np.ndarray] - Grayscale image
    :param show_detailed_steps: [bool] - If 'True' display every step
    :return: [np.ndarray] - Binary image of empty beach

    Otsu thresholding to find out threshold value to use in Canny edge detector (hysteresis).
    Canny edges on empty beach highlight areas that have complicated shapes.
    To connect the area of individual complicated shapes I use morphological closing with large square kernel.
    To close the masked area I added borders to left, right and top of the image.
    Find the largest contour and fill it, then use closing with large elliptic kernel to prevent possible artefacts.
    To account for slight camera/scene shift I use dilation with smaller square kernel.
    """
    thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    canny = cv2.Canny(gray, thresh//2, thresh, (3, 3))

    if show_detailed_steps:
        show_wrap('CANNY EDGES - EMPTY BEACH', canny, 1340)

    kernel = np.ones((31, 31), np.uint8)
    canny_close = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
    canny_close_b = cv2.copyMakeBorder(canny_close, 1, 0, 1, 1, borderType=cv2.BORDER_CONSTANT, value=255)

    if show_detailed_steps:
        show_wrap('CANNY CLOSURE - EMPTY BEACH', canny_close_b, 1340)

    cnts, _ = cv2.findContours(canny_close_b.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

    if show_detailed_steps:
        contour_img = gray.copy()
        cv2.drawContours(contour_img, cnts, -1, (0, 255, 0), 2)
        show_wrap('CONTOUR - EMPTY BEACH', contour_img, 1340)

    empty = np.zeros((gray.shape[0], gray.shape[1]), dtype=np.uint8)
    mask = cv2.fillPoly(empty, pts=[cnts[0]], color=(255, 255, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 23))
    mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    kernel = np.ones((17, 17), np.uint8)
    mask_dilate = cv2.dilate(mask_close, kernel, iterations=1)

    return mask_dilate


def inclusion(box1, box2):
    """
    :param box1: [np.array] - information of bounding box
    :param box2: [np.array] - information of bounding box
    :return: [bool] - 'True' if center of one box is inside the other. 'False' otherwise.

    Test inclusion between two boxes.
    """
    tl1, br1, center1, _ = box1
    tl2, br2, center2, _ = box2

    return ((tl1[0] <= center2[0] <= br1[0] and tl1[1] <= center2[1] <= br1[1]) or
            (tl2[0] <= center1[0] <= br2[0] and tl2[1] <= center1[1] <= br2[1]))


def alignment(box1, box2):
    """
    :param box1: [np.array] - information of bounding box
    :param box2: [np.array] - information of bounding box
    :return: [bool] - 'True' if center of one box is in width of the other and close above. 'False' otherwise.

    Test alignment between two boxes.
    """
    tl1, br1, center1, dim = box1
    tl2, br2, center2, dim = box2
    h_margin = 5
    w_margin = 3

    return ((tl1[0] - w_margin <= center2[0] <= br1[0] + w_margin or
             tl2[0] - w_margin <= center1[0] <= br2[0] + w_margin) and
            ((center1[1] <= center2[1] and tl2[1] <= br1[1] + h_margin) or
             (center1[1] >= center2[1] and tl1[1] <= br2[1] + h_margin)))


def get_all_overlaps(boxes, curr, mode=0):
    """
    :param boxes: [np.array] - Array of all bounding boxes
    :param curr: [int] - Index of box on witch the overlapping is tested
    :param mode: [int] - Switch for different kind of overlaps
    :return: [np.array] - All overlaps with current box

    Test all boxes for overlapping with current box.
    """
    overlaps = []
    for i in range(len(boxes)):
        if i != curr:
            if mode == 0 and inclusion(boxes[curr], boxes[i]):
                overlaps.append(i)
            elif mode == 1 and alignment(boxes[curr], boxes[i]):
                overlaps.append(i)
    return overlaps


def contour_filter(canny, org, show_detailed_steps=False):
    """
    :param canny: [np.ndarray] - Image of canny edges
    :param org: [np.ndarray] - Original image to draw results on
    :param show_detailed_steps: [bool] - If 'True' display every step
    :return: [np.array] - Array of bounding boxes

    Filter through contours based on canny edges.
    Filtering based on size (minimum and maximum) and aspect ratio of the bounding boxes.
    If box fits criteria, add the box information to array.
    Box information contain top left, bottom right and center coordinates, plus width and height.
    """
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    hierarchy = hierarchy[0]
    max_area = 7000
    max_ration = 5
    cnts_filter = org.copy()
    for i in range(len(contours)):
        curr_contour = contours[i]
        curr_hierarchy = hierarchy[i]
        x, y, w, h = cv2.boundingRect(curr_contour)
        if curr_hierarchy[3] < 0:
            if w * h < max_area and w * max_ration > h > 3 and h * max_ration > w > 3:
                cv2.rectangle(cnts_filter, (x, y), (x + w, y + h), (255, 255, 255), 1)
                boxes.append([[x, y], [x + w, y + h], [x + w // 2, y + h // 2], [w, h]])
            else:
                cv2.rectangle(cnts_filter, (x, y), (x + w, y + h), (255, 0, 0), 1)
    if show_detailed_steps:
        show_wrap('CONTOUR FILTERING', cnts_filter, 1340)

    return boxes


def merger(boxes, org, mode=0, show_detailed_steps=False):
    """
    :param boxes: [np.array] - Array of all bounding boxes
    :param org: [np.ndarray] - Original image to draw results on
    :param mode: [int] - Switch for different kind of overlaps
    :param show_detailed_steps: [bool] - If 'True' display every step
    :return: [np.array] - Bounding boxes with no overlaps

    Function that iterates through all boxes and merges overlaps.
    The outer 'while' loop ends when the inner 'while' loop goes through whole array and does not find overlaps.
    Inside inner 'while' loop the overlapping boxes combine into one bigger box, that is put on the front of the
    queue to be tested for overlaps next.
    """
    finished = False
    copy = np.copy(org)
    str_mode = "INCLUDE" if mode == 0 else "ALIGNED"
    if show_detailed_steps:
        print("MODE: " + str_mode)
    while not finished:
        finished = True
        index = len(boxes) - 1
        if show_detailed_steps:
            print("Len Boxes: " + str(len(boxes)))

        while index >= 0:
            overlaps = get_all_overlaps(boxes, index, mode)

            if len(overlaps) > 0:
                con = []
                overlaps.append(index)
                for ind in overlaps:
                    tl, br, _, _ = boxes[ind]
                    con.append([tl])
                    con.append([br])
                con = np.array(con)

                x, y, w, h = cv2.boundingRect(con)
                w -= 1
                h -= 1
                merged = [[x, y], [x+w, y+h], [x+w//2, y+h//2], [w, h]]

                overlaps.sort(reverse=True)
                for ind in overlaps:
                    cv2.rectangle(copy, boxes[ind][0], boxes[ind][1], (0, 0, 255), 1)
                    del boxes[ind]
                boxes.append(merged)

                finished = False
                break

            index -= 1

    if show_detailed_steps:
        for box in boxes:
            cv2.rectangle(copy, box[0], box[1], (255, 0, 0), 1)
        show_wrap("MERGED BOXES - MODE: " + str_mode, copy, 1340)
    return boxes


def find_people_regions(org, gray, mask, show_detailed_steps = False):
    """
    :param org: [np.ndarray] - Original image to draw results on
    :param gray: [np.ndarray] - Gray image to filter through
    :param mask: [np.ndarray] - Mask image to cover unwanted parts of the image
    :param show_detailed_steps: [bool] - If 'True' display every step
    :return: [np.array], [int] - returns array of bounding boxes of interesting areas and otsu threshold value

    This function tries to find people bounding boxes. Because of perspective distortion in these images
    I opted for three levels of blurring. Further the image plain is, the less blurring is desirable.
    On the blured and concatenated picture parts is used canny edge detector and then masked out the part of the image
    that is not of interested. After that the 'contour_filter' and 'merger' sorts through the result and function
    returns possible areas of interest.
    """
    h, w = gray.shape[:2]
    split = h // 20
    persp_gray_1 = cv2.GaussianBlur(gray[split * 15:h][:], (15, 15), 0)
    persp_gray_2 = cv2.GaussianBlur(gray[split * 11:split * 15][:], (11, 11), 0)
    persp_gray_3 = cv2.GaussianBlur(gray[:split * 11][:], (5, 5), 0)

    gray_blur_con = np.concatenate((np.concatenate((persp_gray_3, persp_gray_2), axis=0), persp_gray_1), axis=0)

    thresh, _ = cv2.threshold(gray_blur_con, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    canny = cv2.Canny(gray_blur_con, thresh - 20, thresh + 20, (3, 3))
    canny_masked = cv2.bitwise_and(canny, canny, mask=cv2.bitwise_not(mask))

    if show_detailed_steps:
        show_wrap('persp_gray_1', persp_gray_1, 1340)
        show_wrap('persp_gray_2', persp_gray_2, 1340)
        show_wrap('persp_gray_3', persp_gray_3, 1340)
        show_wrap('gray_blur_con', gray_blur_con, 1340)
        show_wrap('CANNY EDGES - POPULATED BEACH', canny_masked, 1340)

    contour_boxes = contour_filter(canny_masked, np.copy(org), show_detailed_steps)
    merged_boxes = merger(contour_boxes, np.copy(org), 0, show_detailed_steps)

    return merged_boxes, thresh


def filter_small(mrg_sml_boxs, org, show_detailed_steps=False):
    """
    :param mrg_sml_boxs: [np.array] - Array of all bounding boxes
    :param org: [np.ndarray] - Original image to draw results on
    :param show_detailed_steps: [bool] - If 'True' display every step
    :return: [np.array] - Bounding boxes that passed the filters

    Filter out boxes that are too small with respect to perspective.
    """
    filter_boxes = []
    out_boxes = []
    h, w = org.shape[:2]
    copy = np.copy(org)
    split = h // 20
    for b in mrg_sml_boxs:
        _, _, center, dim = b
        if center[1] > 15 * split and (dim[0] < 13 or dim[1] < 13):
            cv2.rectangle(copy, b[0], b[1], (0, 255, 0), 1)
            filter_boxes.append(b)
        elif 15 * split > center[1] > 11 * split and (dim[0] < 8 or dim[1] < 8):
            cv2.rectangle(copy, b[0], b[1], (0, 0, 255), 1)
            filter_boxes.append(b)
        else:
            out_boxes.append(b)
    #if show_detailed_steps:
    print("Number of boxed filtered out" + str(len(filter_boxes)))
    print("Number of output boxes" + str(len(out_boxes)))
    show_wrap("DELETED BOXES", copy, 1340)

    return out_boxes


def clasify_boxes(potential_people_boxes, org, show_detailed_steps=False):
    """
    :param potential_people_boxes: [np.array] - Bounding boxes of potential people or areas with people
    :param org: [np.ndarray] - Original image to draw results on
    :param show_detailed_steps: [bool] - If 'True' display every step
    :return: [np.array, np.array, np.array] - three arrays of small, medium and large areas

    This function should separate differently sized bounding boxes and deal with them accordingly.
    Small boxes - either small objects (persons head) or person split to smaller parts => connect aligned small boxes
    Medium boxes - most likely full body person detected => no need to analyse further
    Large box - too big for one person => analyse further
    """
    sml_area = 300
    big_area = 1500
    copy = np.copy(org)
    sml_boxs = []
    med_boxs = []
    lrg_boxs = []
    for box in potential_people_boxes:
        if box[3][0] * box[3][1] < sml_area:
            sml_boxs.append(box)
            cv2.rectangle(copy, box[0], box[1], (0, 200, 0), 1)
        elif box[3][0] * box[3][1] < big_area:
            med_boxs.append(box)
            cv2.rectangle(copy, box[0], box[1], (200, 0, 0), 1)
        else:
            lrg_boxs.append(box)
            cv2.rectangle(copy, box[0], box[1], (0, 0, 200), 1)
    if show_detailed_steps:
        show_wrap("ALL BOXES", copy, 1340)

    mrg_sml_boxs = merger(sml_boxs, np.copy(org), 1, show_detailed_steps)
    flt_sml_boxs = filter_small(mrg_sml_boxs, org)

    if show_detailed_steps:
        copy_small = np.copy(org)
        for box in flt_sml_boxs:
            cv2.rectangle(copy_small, box[0], box[1], (0, 200, 0), 1)
        show_wrap("SMALL MERGED AND FILTERED", copy_small, 1340)
        copy_medium = np.copy(org)
        for box in med_boxs:
            cv2.rectangle(copy_medium, box[0], box[1], (0, 200, 0), 1)
        show_wrap("MID", copy_medium, 1340)
        copy_large = np.copy(org)
        for box in lrg_boxs:
            cv2.rectangle(copy_large, box[0], box[1], (0, 200, 0), 1)
        show_wrap("LARGE", copy_large, 1340)

    return flt_sml_boxs, med_boxs, lrg_boxs


def print_results(boxes, filename):
    annotations = np.loadtxt('../labels_of_beach_images.csv',
                 delimiter=",", dtype=str)
    count = 0
    for a in annotations:
        if a[5] == filename:
            count += 1
    sml = len(boxes[0])
    mid = len(boxes[1])
    lrg = len(boxes[2])
    total = sml + mid + lrg
    print("Algorithm identified: " + str(total) + " people on the beach.")
    print("      In small boxes: " + str(sml))
    print("     In medium boxes: " + str(mid))
    print("      In large boxes: " + str(lrg))
    print(" Annotated image has: " + str(count) + " people on the beach")


if __name__ == "__main__":
    show_steps = True  # True for 'cv2.imshow()' steps
    show_detailed_steps = False  # True for 'cv2.imshow()' detailed steps
    input_img_id = 8   # values 1 - 9

    img_arr, filenames = load_images_from_folder('../images')
    gray_arr = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in img_arr]
    mask = empty_beach_mask(gray_arr[0], show_detailed_steps)
    gray = gray_arr[input_img_id]

    if show_steps:
        show_wrap('Empty beach mask', mask, 1340)
        org_masked = cv2.bitwise_and(img_arr[input_img_id], img_arr[input_img_id], mask=cv2.bitwise_not(mask))
        show_wrap('Original masked', org_masked, 1340)

    potential_people_boxes, thresh = find_people_regions(img_arr[input_img_id], gray, mask, show_detailed_steps)

    if show_steps:
        copy = np.copy(img_arr[input_img_id])
        for box in potential_people_boxes:
            cv2.rectangle(copy, box[0], box[1], (200, 0, 0), 1)
        show_wrap('Potential people area', copy, 1340)

    sml_boxs, med_boxs, lrg_boxs = clasify_boxes(potential_people_boxes, img_arr[input_img_id], show_detailed_steps)

    if show_steps:
        copy = np.copy(img_arr[input_img_id])
        for box in sml_boxs:
            cv2.rectangle(copy, box[0], box[1], (200, 0, 0), 1)
        for box in med_boxs:
            cv2.rectangle(copy, box[0], box[1], (0, 0, 200), 1)
        for box in lrg_boxs:
            cv2.rectangle(copy, box[0], box[1], (0, 200, 0), 1)
        show_wrap('RESULT', copy, 1340)

    print_results([sml_boxs, med_boxs, lrg_boxs], filenames[input_img_id])
