CALTECH = "C"
LABELME = "L"
PASCAL_VOC = "V"
SUN = "S"

caltech = {
    LABELME: {
        "bird": 12, "car": 178, "chair": 40, "dog": 5, "person": 688,
    },
    PASCAL_VOC: {
        "bird": 9, "car": 9, "chair": 9, "dog": 9, "person": 9,
    },
    SUN: {
        "bird": 7, "car": 7, "chair": 23, "dog": 12, "person": 3,
    },
}
labelme = {
    CALTECH: {
        "bird": 125, "car": 28, "chair": 9, "dog": 3, "person": 483,
    },
    PASCAL_VOC: {
        "bird": 9, "car": 9, "chair": 9, "dog": 9, "person": 9,
    },
    SUN: {
        "bird": 6, "car": 19, "chair": 84, "dog": 12, "person": 2,
    },
}
pascal_voc = {
    CALTECH: {
        "bird": 125, "car": 28, "chair": 9, "dog": 3, "person": 483,
    },
    LABELME: {
        "bird": 1, "car": 3, "chair": 13, "dog": 22, "person": 1,
    },
    SUN: {
        "bird": 9, "car": 10, "chair": 10, "dog": 10, "person": 10,
    },
}
sun = {
    CALTECH: {
        "bird": 125, "car": 28, "chair": 9, "dog": 3, "person": 483,
    },
    LABELME: {
        "bird": 1, "car": 6, "chair": 26, "dog": 19, "person": 1,
    },
    PASCAL_VOC: {
        "bird": 8, "car": 8, "chair": 8, "dog": 8, "person": 8,
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
        "many": 100, "few": 20,
    },
    LABELME: {
        "many": 100, "few": 20,
    },
    PASCAL_VOC: {
        "many": 100, "few": 20,
    },
    SUN: {
        "many": 100, "few": 20,
    },
}

validation = {
    CALTECH: 5,
    LABELME: 5,
    PASCAL_VOC: 5,
    SUN: 5,
}

