ART_PAINTING = "A"
CARTOON = "C"
PHOTO = "P"
SKETCH = "S"

art_painting = {
    CARTOON: {
        "dog": 29, "elephant": 5, "giraffe": 197, "guitar": 89,
        "horse": 11, "house": 2, "person": 1,
    },
    PHOTO: {
        "dog": 8, "elephant": 50, "giraffe": 2, "guitar": 4,
        "horse": 19, "house": 150, "person": 333,
    },
    SKETCH: {
        "dog": 6, "elephant": 6, "giraffe": 6, "guitar": 6,
        "horse": 6, "house": 6, "person": 6,
    },
}
cartoon = {
    ART_PAINTING: {
        "dog": 33, "elephant": 5, "giraffe": 216, "guitar": 101,
        "horse": 13, "house": 3, "person": 1,
    },
    PHOTO: {
        "dog": 8, "elephant": 50, "giraffe": 2, "guitar": 4,
        "horse": 19, "house": 150, "person": 333,
    },
    SKETCH: {
        "dog": 7, "elephant": 7, "giraffe": 7, "guitar": 7,
        "horse": 7, "house": 7, "person": 7,
    },
}
photo = {
    ART_PAINTING: {
        "dog": 156, "elephant": 9, "giraffe": 20, "guitar": 2,
        "horse": 4, "house": 52, "person": 347,
    },
    CARTOON: {
        "dog": 1, "elephant": 5, "giraffe": 2, "guitar": 35,
        "horse": 16, "house": 1, "person": 1,
    },
    SKETCH: {
        "dog": 8, "elephant": 8, "giraffe": 8, "guitar": 8,
        "horse": 8, "house": 8, "person": 8,
    },
}
sketch = {
    ART_PAINTING: {
        "dog": 11, "elephant": 1, "giraffe": 73, "guitar": 33,
        "horse": 4, "house": 1, "person": 1,
    },
    CARTOON: {
        "dog": 6, "elephant": 6, "giraffe": 6, "guitar": 6,
        "horse": 6, "house": 6, "person": 6,
    },
    PHOTO: {
        "dog": 8, "elephant": 50, "giraffe": 2, "guitar": 4,
        "horse": 19, "house": 150, "person": 333,
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
        "many": 100, "few": 30,
    },
    CARTOON: {
        "many": 100, "few": 30,
    },
    PHOTO: {
        "many": 100, "few": 15,
    },
    SKETCH: {
        "many": 100, "few": 15,
    },
}

validation = {
    ART_PAINTING: 15,
    CARTOON: 15,
    PHOTO: 15,
    SKETCH: 15,
}

