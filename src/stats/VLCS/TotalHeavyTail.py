CALTECH = "C"
LABELME = "L"
PASCAL_VOC = "V"
SUN = "S"

caltech = {
    LABELME: {
        "bird": 3, "car": 246, "chair": 6, "dog": 3, "person": 688,
    },
    PASCAL_VOC: {
        "bird": 13, "car": 141, "chair": 36, "dog": 36, "person": 835,
    },
    SUN: {
        "bird": 1, "car": 189, "chair": 88, "dog": 2, "person": 704,
    },
}
labelme = {
    CALTECH: {
        "bird": 15, "car": 35, "chair": 8, "dog": 1, "person": 483,
    },
    PASCAL_VOC: {
        "bird": 21, "car": 209, "chair": 32, "dog": 12, "person": 835,
    },
    SUN: {
        "bird": 1, "car": 279, "chair": 78, "dog": 1, "person": 704,
    },
}
pascal_voc = {
    CALTECH: {
        "bird": 25, "car": 25, "chair": 10, "dog": 7, "person": 483,
    },
    LABELME: {
        "bird": 8, "car": 260, "chair": 7, "dog": 4, "person": 688,
    },
    SUN: {
        "bird": 1, "car": 200, "chair": 93, "dog": 2, "person": 704,
    },
}
sun = {
    CALTECH: {
        "bird": 43, "car": 29, "chair": 6, "dog": 1, "person": 483,
    },
    LABELME: {
        "bird": 13, "car": 310, "chair": 5, "dog": 1, "person": 688,
    },
    PASCAL_VOC: {
        "bird": 60, "car": 178, "chair": 25, "dog": 12, "person": 835,
    },
}
mapping = {
    CALTECH: caltech,
    LABELME: labelme,
    PASCAL_VOC: pascal_voc,
    SUN: sun,
}

thres = {
    CALTECH: {
        "many": 200, "few": 14,
    },
    LABELME: {
        "many": 200, "few": 14,
    },
    PASCAL_VOC: {
        "many": 200, "few": 14,
    },
    SUN: {
        "many": 200, "few": 14,
    },
}

validation = {
    CALTECH: 5,
    LABELME: 5,
    PASCAL_VOC: 5,
    SUN: 5,
}

