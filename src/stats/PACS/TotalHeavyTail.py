ART_PAINTING = "A"
CARTOON = "C"
PHOTO = "P"
SKETCH = "S"

art_painting = {
    CARTOON: {
        "dog": 140, "elephant": 353, "giraffe": 17, "guitar": 2,
        "horse": 39, "house": 3, "person": 11,
    },
    PHOTO: {
        "dog": 65, "elephant": 149, "giraffe": 8, "guitar": 2,
        "horse": 23, "house": 3, "person": 12,
    },
    SKETCH: {
        "dog": 283, "elephant": 580, "giraffe": 38, "guitar": 9,
        "horse": 101, "house": 1, "person": 4,
    },
}
cartoon = {
    ART_PAINTING: {
        "dog": 291, "elephant": 12, "giraffe": 107, "guitar": 2,
        "horse": 24, "house": 3, "person": 12,
    },
    PHOTO: {
        "dog": 139, "elephant": 9, "giraffe": 66, "guitar": 2,
        "horse": 24, "house": 3, "person": 11,
    },
    SKETCH: {
        "dog": 605, "elephant": 38, "giraffe": 293, "guitar": 8,
        "horse": 107, "house": 1, "person": 4,
    },
}
photo = {
    ART_PAINTING: {
        "dog": 291, "elephant": 92, "giraffe": 36, "guitar": 3,
        "horse": 10, "house": 4, "person": 14,
    },
    CARTOON: {
        "dog": 299, "elephant": 169, "giraffe": 44, "guitar": 2,
        "horse": 16, "house": 4, "person": 12,
    },
    SKETCH: {
        "dog": 605, "elephant": 278, "giraffe": 100, "guitar": 10,
        "horse": 43, "house": 1, "person": 4,
    },
}
sketch = {
    ART_PAINTING: {
        "dog": 178, "elephant": 41, "giraffe": 9, "guitar": 2,
        "horse": 3, "house": 20, "person": 347,
    },
    CARTOON: {
        "dog": 183, "elephant": 76, "giraffe": 11, "guitar": 1,
        "horse": 6, "house": 19, "person": 312,
    },
    PHOTO: {
        "dog": 85, "elephant": 32, "giraffe": 5, "guitar": 2,
        "horse": 3, "house": 18, "person": 333,
    },
}
mapping = {
    ART_PAINTING: art_painting,
    CARTOON: cartoon,
    PHOTO: photo,
    SKETCH: sketch,
}

thres = {
    ART_PAINTING: {
        "many": 120, "few": 6,
    },
    CARTOON: {
        "many": 120, "few": 6,
    },
    PHOTO: {
        "many": 120, "few": 6,
    },
    SKETCH: {
        "many": 120, "few": 6,
    },
}

validation = {
    ART_PAINTING: 15,
    CARTOON: 15,
    PHOTO: 15,
    SKETCH: 15,
}

