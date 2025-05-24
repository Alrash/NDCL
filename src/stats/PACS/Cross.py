ART_PAINTING = "A"
CARTOON = "C"
PHOTO = "P"
SKETCH = "S"

art_painting = {
    CARTOON: {
        "dog": 53, "elephant": 353, "giraffe": 20, "guitar": 2,
        "horse": 9, "house": 4, "person": 159,
    },
    PHOTO: {
        "dog": 26, "elephant": 26, "giraffe": 26, "guitar": 26,
        "horse": 26, "house": 26, "person": 26,
    },
    SKETCH: {
        "dog": 12, "elephant": 3, "giraffe": 27, "guitar": 474,
        "horse": 71, "house": 52, "person": 6,
    },
}
cartoon = {
    ART_PAINTING: {
        "dog": 156, "elephant": 9, "giraffe": 20, "guitar": 2,
        "horse": 4, "house": 52, "person": 347,
    },
    PHOTO: {
        "dog": 29, "elephant": 29, "giraffe": 29, "guitar": 29,
        "horse": 29, "house": 29, "person": 29,
    },
    SKETCH: {
        "dog": 6, "elephant": 71, "giraffe": 27, "guitar": 474,
        "horse": 214, "house": 12, "person": 3,
    },
}
photo = {
    ART_PAINTING: {
        "dog": 26, "elephant": 26, "giraffe": 26, "guitar": 26,
        "horse": 26, "house": 26, "person": 26,
    },
    CARTOON: {
        "dog": 53, "elephant": 353, "giraffe": 20, "guitar": 2,
        "horse": 9, "house": 4, "person": 159,
    },
    SKETCH: {
        "dog": 12, "elephant": 3, "giraffe": 27, "guitar": 474,
        "horse": 71, "house": 52, "person": 6,
    },
}
sketch = {
    ART_PAINTING: {
        "dog": 156, "elephant": 9, "giraffe": 20, "guitar": 2,
        "horse": 4, "house": 52, "person": 347,
    },
    CARTOON: {
        "dog": 1, "elephant": 14, "giraffe": 5, "guitar": 95,
        "horse": 43, "house": 2, "person": 1,
    },
    PHOTO: {
        "dog": 16, "elephant": 16, "giraffe": 16, "guitar": 16,
        "horse": 16, "house": 16, "person": 16,
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
        "many": 160, "few": 60,
    },
    CARTOON: {
        "many": 160, "few": 60,
    },
    PHOTO: {
        "many": 160, "few": 60,
    },
    SKETCH: {
        "many": 120, "few": 30,
    },
}

validation = {
    ART_PAINTING: 15,
    CARTOON: 15,
    PHOTO: 15,
    SKETCH: 15,
}

