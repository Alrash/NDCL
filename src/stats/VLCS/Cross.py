CALTECH = "C"
LABELME = "L"
PASCAL_VOC = "V"
SUN = "S"

caltech = {
    LABELME: {
        "bird": 19, "car": 19, "chair": 19, "dog": 19, "person": 19,
    },
    PASCAL_VOC: {
        "bird": 180, "car": 10, "chair": 3, "dog": 46, "person": 1,
    },
    SUN: {
        "bird": 5, "car": 41, "chair": 182, "dog": 12, "person": 704,
    },
}
labelme = {
    CALTECH: {
        "bird": 33, "car": 33, "chair": 33, "dog": 33, "person": 33,
    },
    PASCAL_VOC: {
        "bird": 180, "car": 10, "chair": 3, "dog": 46, "person": 1,
    },
    SUN: {
        "bird": 5, "car": 41, "chair": 182, "dog": 12, "person": 704,
    },
}
pascal_voc = {
    CALTECH: {
        "bird": 28, "car": 28, "chair": 28, "dog": 28, "person": 28,
    },
    LABELME: {
        "bird": 12, "car": 178, "chair": 40, "dog": 5, "person": 688,
    },
    SUN: {
        "bird": 3, "car": 1, "chair": 5, "dog": 12, "person": 19,
    },
}
sun = {
    CALTECH: {
        "bird": 33, "car": 33, "chair": 33, "dog": 33, "person": 33,
    },
    LABELME: {
        "bird": 12, "car": 178, "chair": 40, "dog": 5, "person": 688,
    },
    PASCAL_VOC: {
        "bird": 59, "car": 4, "chair": 13, "dog": 231, "person": 1,
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
        "many": 200, "few": 40,
    },
    LABELME: {
        "many": 200, "few": 40,
    },
    PASCAL_VOC: {
        "many": 200, "few": 40,
    },
    SUN: {
        "many": 200, "few": 40,
    },
}

validation = {
    CALTECH: 5,
    LABELME: 5,
    PASCAL_VOC: 5,
    SUN: 5,
}

