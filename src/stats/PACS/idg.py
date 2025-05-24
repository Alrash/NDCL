ART_PAINTING = 'A'
CARTOON = 'C'
PHOTO = 'P'
SKETCH = 'S'

art_painting = {
    CARTOON: {
        'dog': 350, 'elephant': 327, 'giraffe': 216,
        'guitar': 87, 'horse': 97, 'house': 89, 'person': 41
    },
    PHOTO: {
        'dog': 157, 'elephant': 144, 'giraffe': 114,
        'guitar': 120, 'horse': 59, 'house': 86, 'person': 43
    },
    SKETCH: {
        'dog': 743, 'elephant': 529, 'giraffe': 470, 'guitar': 393,
        'horse': 244, 'house': 25, 'person': 16
    }
}

cartoon = {
    ART_PAINTING: {
        'dog': 343, 'elephant': 213, 'giraffe': 187,
        'guitar': 113, 'horse': 66, 'house': 90, 'person': 43
    },
    PHOTO: {
        'dog': 157, 'elephant': 169, 'giraffe': 119,
        'guitar': 114, 'horse': 65, 'house': 85, 'person': 41
    },
    SKETCH: {
        'dog': 724, 'elephant': 618, 'giraffe': 494,
        'guitar': 373, 'horse': 268, 'house': 24, 'person': 15
    }
}

photo = {
    ART_PAINTING: {
        'dog': 320, 'elephant': 176, 'giraffe': 165,
        'guitar': 119, 'horse': 60, 'house': 89, 'person': 44
    },
    CARTOON: {
        'dog': 328, 'elephant': 315, 'giraffe': 200,
        'guitar': 87, 'horse': 97, 'house': 87, 'person': 40
    },
    SKETCH: {
        'dog': 652, 'elephant': 510, 'giraffe': 435,
        'guitar': 394, 'horse': 243, 'house': 24, 'person': 16
    }
}

sketch = {
    ART_PAINTING: {
        'dog': 331, 'elephant': 223, 'giraffe': 210,
        'guitar': 146, 'horse': 56, 'house': 34, 'person': 17
    },
    CARTOON: {
        'dog': 341, 'elephant': 400, 'giraffe': 255,
        'guitar': 107, 'horse': 90, 'house': 33, 'person': 16
    },
    PHOTO: {
        'dog': 153, 'elephant': 152, 'giraffe': 134,
        'guitar': 147, 'horse': 55, 'house': 32, 'person': 17
    }
}


mapping = {
    ART_PAINTING: art_painting,
    CARTOON: cartoon,
    PHOTO: photo,
    SKETCH: sketch,
}

thres = {
    ART_PAINTING: {'many': 200, 'few': 50},
    CARTOON: {'many': 200, 'few': 50},
    PHOTO: {'many': 200, 'few': 50},
    SKETCH: {'many': 200, 'few': 50},
}

validation = {
    ART_PAINTING: 25,
    CARTOON: 25,
    PHOTO: 25,
    SKETCH: 25,
}
