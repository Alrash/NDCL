CALTECH = 'C'
LABELME = 'L'
SUN = 'S'
VOC = 'V'


caltech = {
    LABELME: {'bird': 37, 'car': 426, 'chair': 34, 'dog': 20, 'person': 619},
    SUN: {'bird': 7, 'car': 328, 'chair': 400, 'dog': 13, 'person': 632},
    VOC: {'bird': 153, 'car': 246, 'chair': 166, 'dog': 256, 'person': 749},
}


labelme = {
    CALTECH: {'bird': 81, 'car': 70, 'chair': 45, 'dog': 34, 'person': 479},
    SUN: {'bird': 7, 'car': 531, 'chair': 393, 'dog': 12, 'person': 696},
    VOC: {'bird': 112, 'car': 399, 'chair': 162, 'dog': 244, 'person': 825},
}

voc = {
    CALTECH: {'bird': 141, 'car': 54, 'chair': 57, 'dog': 39, 'person': 516},
    LABELME: {'bird': 42, 'car': 534, 'chair': 43, 'dog': 22, 'person': 734},
    SUN: {'bird': 7, 'car': 412, 'chair': 500, 'dog': 14, 'person': 750},
}

sun = {
    CALTECH: {'bird': 147, 'car': 61, 'chair': 56, 'dog': 25, 'person': 483},
    LABELME: {'bird': 44, 'car': 595, 'chair': 42, 'dog': 16, 'person': 686},
    VOC: {'bird': 204, 'car': 344, 'chair': 202, 'dog': 159, 'person': 831},
}


mapping = {
    CALTECH: caltech,
    LABELME: labelme,
    SUN: sun,
    VOC: voc,
}

thres = {
    CALTECH: {'many': 250, 'few': 100},
    LABELME: {'many': 250, 'few': 100},
    SUN: {'many': 250, 'few': 100},
    VOC: {'many': 250, 'few': 100},
}


validation = {
    CALTECH: 5,
    LABELME: 5,
    SUN: 5,
    VOC: 5,
}

# import numpy as np
# for key in mapping.keys():
#     domains = list(mapping[key].keys())
#     classes = list(mapping[key][domains[0]].keys())
#     counter = np.zeros(len(classes))
#     for d in domains:
#         for k in classes:
#             counter[classes.index(k)] += mapping[key][d][k]
#     counter /= 3
#     print(counter)