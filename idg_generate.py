import math
import os
import sys
import copy
import numpy as np
from scipy.stats import fisk
import collections, functools, operator
from matplotlib import pyplot as plt
from typing import List, Union


seed = 20240101
np.random.seed(seed)


DATASETS = {
    'pacs': 'PACS',
    'vlcs': 'VLCS',
    'officehome': 'OfficeHome',
    'domainnet': 'DomainNet',
}

DOMAINS = {
    DATASETS['pacs']: {
        'art_painting': 'A',
        'cartoon': 'C',
        'photo': 'P',
        'sketch': 'S',
    },
    DATASETS['vlcs']: {
        'caltech': 'C',
        'labelme': 'L',
        'pascal_voc': 'V',
        'sun': 'S',
    },
    DATASETS['officehome']: {
        'art': 'A',
        'clipart': 'C',
        'product': 'P',
        'real_world': 'R',
    },
    DATASETS['domainnet']: {
        'clipart': 'clip',
        'infograph': 'info',
        'painting': 'paint',
        'quickdraw': 'quick',
        'real': 'real',
        'sketch': 'sketch',
    }
}

STATS = {
    DATASETS['pacs']: {
        'art_painting': {'dog': 379, 'elephant': 255, 'giraffe': 285, 'guitar': 184, 'horse': 201, 'house': 295, 'person': 449},
        'cartoon': {'dog': 389, 'elephant': 457, 'giraffe': 346, 'guitar': 135, 'horse': 324, 'house': 288, 'person': 405},
        'photo': {'dog': 189, 'elephant': 202, 'giraffe': 182, 'guitar': 186, 'horse': 199, 'house': 280, 'person': 432},
        'sketch': {'dog': 772, 'elephant': 740, 'giraffe': 753, 'guitar': 608, 'horse': 816, 'house': 80, 'person': 160},
    },
    DATASETS['vlcs']: {
        'caltech': {'bird': 166, 'car': 86, 'chair': 83, 'dog': 47, 'person': 609},
        'labelme': {'bird': 56, 'car': 846, 'chair': 62, 'dog': 29, 'person': 866},
        'pascal_voc': {'bird': 231, 'car': 489, 'chair': 300, 'dog': 294, 'person': 1049},
        'sun': {'bird': 14, 'car': 652, 'chair': 725, 'dog': 21, 'person': 885},
    },
    DATASETS['officehome']: {
        'art': {
            'Alarm_Clock': 74, 'Backpack': 41, 'Batteries': 27, 'Bed': 40, 'Bike': 75, 'Bottle': 99, 'Bucket': 40, 'Calculator': 33, 'Calendar': 20, 'Candles': 76,
            'Chair': 69, 'Clipboards': 25, 'Computer': 44, 'Couch': 40, 'Curtains': 40, 'Desk_Lamp': 23, 'Drill': 15, 'Eraser': 18, 'Exit_Sign': 21, 'Fan': 45,
            'File_Cabinet': 22, 'Flipflops': 46, 'Flowers': 90, 'Folder': 20, 'Fork': 46, 'Glasses': 40, 'Hammer': 40, 'Helmet': 79, 'Kettle': 46, 'Keyboard': 18,
            'Knives': 72, 'Lamp_Shade': 49, 'Laptop': 51, 'Marker': 20, 'Monitor': 42, 'Mop': 32, 'Mouse': 18, 'Mug': 49, 'Notebook': 21, 'Oven': 20,
            'Pan': 19, 'Paper_Clip': 19, 'Pen': 20, 'Pencil': 26, 'Postit_Notes': 19, 'Printer': 18, 'Push_Pin': 24, 'Radio': 47, 'Refrigerator': 49, 'Ruler': 15,
            'Scissors': 20, 'Screwdriver': 30, 'Shelf': 42, 'Sink': 41, 'Sneakers': 46, 'Soda': 40, 'Speaker': 20, 'Spoon': 46, 'TV': 40, 'Table': 16,
            'Telephone': 44, 'ToothBrush': 43, 'Toys': 20, 'Trash_Can': 21, 'Webcam': 16
        },
        'clipart': {
            'Alarm_Clock': 60, 'Backpack': 56, 'Batteries': 64, 'Bed': 98, 'Bike': 99, 'Bottle': 99, 'Bucket': 73, 'Calculator': 46, 'Calendar': 78, 'Candles': 99,
            'Chair': 99, 'Clipboards': 40, 'Computer': 99, 'Couch': 64, 'Curtains': 42, 'Desk_Lamp': 41, 'Drill': 48, 'Eraser': 40, 'Exit_Sign': 41, 'Fan': 50,
            'File_Cabinet': 40, 'Flipflops': 40, 'Flowers': 99, 'Folder': 99, 'Fork': 61, 'Glasses': 52, 'Hammer': 99, 'Helmet': 69, 'Kettle': 40, 'Keyboard': 99,
            'Knives': 53, 'Lamp_Shade': 40, 'Laptop': 99, 'Marker': 71, 'Monitor': 99, 'Mop': 40, 'Mouse': 76, 'Mug': 99, 'Notebook': 83, 'Oven': 40,
            'Pan': 51, 'Paper_Clip': 40, 'Pen': 99, 'Pencil': 99, 'Postit_Notes': 41, 'Printer': 87, 'Push_Pin': 40, 'Radio': 46, 'Refrigerator': 40, 'Ruler': 67,
            'Scissors': 99, 'Screwdriver': 75, 'Shelf': 42, 'Sink': 42, 'Sneakers': 43, 'Soda': 61, 'Speaker': 90, 'Spoon': 60, 'TV': 99, 'Table': 80,
            'Telephone': 99, 'ToothBrush': 39, 'Toys': 99, 'Trash_Can': 53, 'Webcam': 40
        },
        'product': {
            'Alarm_Clock': 79, 'Backpack': 99, 'Batteries': 62, 'Bed': 43, 'Bike': 44, 'Bottle': 62, 'Bucket': 47, 'Calculator': 81, 'Calendar': 81, 'Candles': 56,
            'Chair': 99, 'Clipboards': 65, 'Computer': 96, 'Couch': 88, 'Curtains': 75, 'Desk_Lamp': 83, 'Drill': 67, 'Eraser': 41, 'Exit_Sign': 67, 'Fan': 58,
            'File_Cabinet': 71, 'Flipflops': 99, 'Flowers': 91, 'Folder': 90, 'Fork': 41, 'Glasses': 67, 'Hammer': 57, 'Helmet': 90, 'Kettle': 72, 'Keyboard': 99,
            'Knives': 41, 'Lamp_Shade': 54, 'Laptop': 99, 'Marker': 56, 'Monitor': 98, 'Mop': 72, 'Mouse': 96, 'Mug': 41, 'Notebook': 93, 'Oven': 68,
            'Pan': 70, 'Paper_Clip': 47, 'Pen': 60, 'Pencil': 40, 'Postit_Notes': 38, 'Printer': 99, 'Push_Pin': 43, 'Radio': 43, 'Refrigerator': 59, 'Ruler': 58,
            'Scissors': 99, 'Screwdriver': 40, 'Shelf': 49, 'Sink': 46, 'Sneakers': 99, 'Soda': 43, 'Speaker': 99, 'Spoon': 47, 'TV': 76, 'Table': 60,
            'Telephone': 58, 'ToothBrush': 42, 'Toys': 45, 'Trash_Can': 93, 'Webcam': 98
        },
        'real_world': {
            'Alarm_Clock': 86, 'Backpack': 99, 'Batteries': 64, 'Bed': 83, 'Bike': 99, 'Bottle': 78, 'Bucket': 80, 'Calculator': 73, 'Calendar': 68, 'Candles': 99,
            'Chair': 96, 'Clipboards': 65, 'Computer': 64, 'Couch': 76, 'Curtains': 73, 'Desk_Lamp': 62, 'Drill': 51, 'Eraser': 43, 'Exit_Sign': 81, 'Fan': 60,
            'File_Cabinet': 58, 'Flipflops': 85, 'Flowers': 75, 'Folder': 57, 'Fork': 36, 'Glasses': 60, 'Hammer': 52, 'Helmet': 60, 'Kettle': 72, 'Keyboard': 75,
            'Knives': 83, 'Lamp_Shade': 78, 'Laptop': 67, 'Marker': 23, 'Monitor': 71, 'Mop': 46, 'Mouse': 60, 'Mug': 58, 'Notebook': 68, 'Oven': 64,
            'Pan': 30, 'Paper_Clip': 68, 'Pen': 65, 'Pencil': 59, 'Postit_Notes': 67, 'Printer': 52, 'Push_Pin': 53, 'Radio': 66, 'Refrigerator': 75, 'Ruler': 41,
            'Scissors': 77, 'Screwdriver': 51, 'Shelf': 66, 'Sink': 77, 'Sneakers': 88, 'Soda': 63, 'Speaker': 81, 'Spoon': 54, 'TV': 53, 'Table': 59,
            'Telephone': 82, 'ToothBrush': 85, 'Toys': 67, 'Trash_Can': 81, 'Webcam': 49
        },
    },
    DATASETS['domainnet']: {
        'clipart': {
            'The_Eiffel_Tower': 114, 'The_Great_Wall_of_China': 116, 'The_Mona_Lisa': 150, 'aircraft_carrier': 27, 'airplane': 73, 'alarm_clock': 93, 'ambulance': 80, 'angel': 165, 'animal_migration': 235, 'ant': 81,
            'anvil': 122, 'apple': 88, 'arm': 50, 'asparagus': 49, 'axe': 48, 'backpack': 33, 'banana': 50, 'bandage': 47, 'barn': 157, 'baseball': 116,
            'baseball_bat': 106, 'basket': 78, 'basketball': 61, 'bat': 35, 'bathtub': 100, 'beach': 105, 'bear': 124, 'beard': 156, 'bed': 197, 'bee': 202,
            'belt': 137, 'bench': 71, 'bicycle': 71, 'binoculars': 246, 'bird': 336, 'birthday_cake': 165, 'blackberry': 106, 'blueberry': 171, 'book': 167, 'boomerang': 92,
            'bottlecap': 118, 'bowtie': 146, 'bracelet': 293, 'brain': 76, 'bread': 197, 'bridge': 66, 'broccoli': 105, 'broom': 171, 'bucket': 142, 'bulldozer': 111,
            'bus': 101, 'bush': 46, 'butterfly': 160, 'cactus': 119, 'cake': 108, 'calculator': 55, 'calendar': 66, 'camel': 154, 'camera': 58, 'camouflage': 181,
            'campfire': 122, 'candle': 151, 'cannon': 103, 'canoe': 68, 'car': 99, 'carrot': 52, 'castle': 47, 'cat': 43, 'ceiling_fan': 35, 'cell_phone': 38,
            'cello': 93, 'chair': 94, 'chandelier': 223, 'church': 54, 'circle': 199, 'clarinet': 214, 'clock': 69, 'cloud': 172, 'coffee_cup': 357, 'compass': 191,
            'computer': 287, 'cookie': 97, 'cooler': 214, 'couch': 232, 'cow': 188, 'crab': 108, 'crayon': 141, 'crocodile': 164, 'crown': 208, 'cruise_ship': 208,
            'cup': 128, 'diamond': 207, 'dishwasher': 109, 'diving_board': 182, 'dog': 70, 'dolphin': 84, 'donut': 139, 'door': 81, 'dragon': 105, 'dresser': 41,
            'drill': 136, 'drums': 194, 'duck': 142, 'dumbbell': 387, 'ear': 101, 'elbow': 199, 'elephant': 115, 'envelope': 147, 'eraser': 138, 'eye': 108,
            'eyeglasses': 201, 'face': 54, 'fan': 148, 'feather': 268, 'fence': 165, 'finger': 153, 'fire_hydrant': 149, 'fireplace': 138, 'firetruck': 167, 'fish': 130,
            'flamingo': 274, 'flashlight': 221, 'flip_flops': 147, 'floor_lamp': 180, 'flower': 253, 'flying_saucer': 233, 'foot': 85, 'fork': 200, 'frog': 163, 'frying_pan': 187,
            'garden': 63, 'garden_hose': 147, 'giraffe': 134, 'goatee': 255, 'golf_club': 207, 'grapes': 93, 'grass': 148, 'guitar': 103, 'hamburger': 187, 'hammer': 147,
            'hand': 97, 'harp': 258, 'hat': 120, 'headphones': 285, 'hedgehog': 138, 'helicopter': 145, 'helmet': 163, 'hexagon': 196, 'hockey_puck': 188, 'hockey_stick': 155,
            'horse': 201, 'hospital': 95, 'hot_air_balloon': 198, 'hot_dog': 38, 'hot_tub': 144, 'hourglass': 100, 'house': 108, 'house_plant': 25, 'hurricane': 92, 'ice_cream': 160,
            'jacket': 72, 'jail': 104, 'kangaroo': 106, 'key': 59, 'keyboard': 32, 'knee': 45, 'knife': 32, 'ladder': 74, 'lantern': 179, 'laptop': 26,
            'leaf': 12, 'leg': 89, 'light_bulb': 12, 'lighter': 66, 'lighthouse': 123, 'lightning': 171, 'line': 13, 'lion': 46, 'lipstick': 101, 'lobster': 243,
            'lollipop': 74, 'mailbox': 18, 'map': 42, 'marker': 57, 'matches': 60, 'megaphone': 73, 'mermaid': 207, 'microphone': 143, 'microwave': 16, 'monkey': 123,
            'moon': 126, 'mosquito': 56, 'motorbike': 42, 'mountain': 67, 'mouse': 74, 'moustache': 222, 'mouth': 110, 'mug': 168, 'mushroom': 136, 'nail': 41,
            'necklace': 83, 'nose': 57, 'ocean': 54, 'octagon': 29, 'octopus': 190, 'onion': 87, 'oven': 14, 'owl': 133, 'paint_can': 60, 'paintbrush': 25,
            'palm_tree': 65, 'panda': 87, 'pants': 16, 'paper_clip': 122, 'parachute': 82, 'parrot': 75, 'passport': 26, 'peanut': 84, 'pear': 74, 'peas': 90,
            'pencil': 51, 'penguin': 121, 'piano': 20, 'pickup_truck': 46, 'picture_frame': 88, 'pig': 93, 'pillow': 151, 'pineapple': 83, 'pizza': 77, 'pliers': 38,
            'police_car': 104, 'pond': 105, 'pool': 139, 'popsicle': 288, 'postcard': 91, 'potato': 86, 'power_outlet': 25, 'purse': 41, 'rabbit': 105, 'raccoon': 187,
            'radio': 30, 'rain': 71, 'rainbow': 61, 'rake': 119, 'remote_control': 117, 'rhinoceros': 102, 'rifle': 83, 'river': 134, 'roller_coaster': 143, 'rollerskates': 204,
            'sailboat': 162, 'sandwich': 189, 'saw': 76, 'saxophone': 236, 'school_bus': 230, 'scissors': 205, 'scorpion': 171, 'screwdriver': 205, 'sea_turtle': 236, 'see_saw': 299,
            'shark': 203, 'sheep': 114, 'shoe': 127, 'shorts': 140, 'shovel': 214, 'sink': 133, 'skateboard': 263, 'skull': 178, 'skyscraper': 195, 'sleeping_bag': 96,
            'smiley_face': 113, 'snail': 166, 'snake': 168, 'snorkel': 278, 'snowflake': 175, 'snowman': 174, 'soccer_ball': 85, 'sock': 167, 'speedboat': 271, 'spider': 161,
            'spoon': 228, 'spreadsheet': 187, 'square': 163, 'squiggle': 148, 'squirrel': 221, 'stairs': 386, 'star': 111, 'steak': 155, 'stereo': 289, 'stethoscope': 343,
            'stitches': 206, 'stop_sign': 169, 'stove': 256, 'strawberry': 357, 'streetlight': 326, 'string_bean': 139, 'submarine': 344, 'suitcase': 377, 'sun': 248, 'swan': 469,
            'sweater': 222, 'swing_set': 143, 'sword': 139, 'syringe': 128, 't-shirt': 98, 'table': 297, 'teapot': 222, 'teddy-bear': 124, 'telephone': 148, 'television': 136,
            'tennis_racquet': 187, 'tent': 153, 'tiger': 315, 'toaster': 196, 'toe': 85, 'toilet': 175, 'tooth': 101, 'toothbrush': 159, 'toothpaste': 105, 'tornado': 169,
            'tractor': 154, 'traffic_light': 211, 'train': 109, 'tree': 126, 'triangle': 183, 'trombone': 227, 'truck': 117, 'trumpet': 117, 'umbrella': 145, 'underwear': 253,
            'van': 207, 'vase': 161, 'violin': 174, 'washing_machine': 265, 'watermelon': 193, 'waterslide': 159, 'whale': 343, 'wheel': 133, 'windmill': 245, 'wine_bottle': 230,
            'wine_glass': 220, 'wristwatch': 285, 'yoga': 165, 'zebra': 235, 'zigzag': 323,
        },
        'infograph': {
            'The_Eiffel_Tower': 190, 'The_Great_Wall_of_China': 80, 'The_Mona_Lisa': 112, 'aircraft_carrier': 88, 'airplane': 62, 'alarm_clock': 23, 'ambulance': 20, 'angel': 17, 'animal_migration': 68, 'ant': 62,
            'anvil': 23, 'apple': 75, 'arm': 129, 'asparagus': 134, 'axe': 92, 'backpack': 341, 'banana': 376, 'bandage': 322, 'barn': 150, 'baseball': 369,
            'baseball_bat': 353, 'basket': 219, 'basketball': 219, 'bat': 99, 'bathtub': 135, 'beach': 183, 'bear': 81, 'beard': 93, 'bed': 180, 'bee': 233,
            'belt': 95, 'bench': 47, 'bicycle': 272, 'binoculars': 55, 'bird': 208, 'birthday_cake': 69, 'blackberry': 214, 'blueberry': 110, 'book': 188, 'boomerang': 45,
            'bottlecap': 26, 'bowtie': 95, 'bracelet': 123, 'brain': 283, 'bread': 232, 'bridge': 61, 'broccoli': 229, 'broom': 35, 'bucket': 56, 'bulldozer': 55,
            'bus': 183, 'bush': 12, 'butterfly': 162, 'cactus': 36, 'cake': 140, 'calculator': 28, 'calendar': 54, 'camel': 31, 'camera': 66, 'camouflage': 27,
            'campfire': 53, 'candle': 68, 'cannon': 14, 'canoe': 71, 'car': 356, 'carrot': 251, 'castle': 123, 'cat': 172, 'ceiling_fan': 63, 'cell_phone': 170,
            'cello': 34, 'chair': 148, 'chandelier': 29, 'church': 20, 'circle': 248, 'clarinet': 25, 'clock': 50, 'cloud': 142, 'coffee_cup': 191, 'compass': 36,
            'computer': 97, 'cookie': 78, 'cooler': 21, 'couch': 61, 'cow': 134, 'crab': 50, 'crayon': 41, 'crocodile': 56, 'crown': 17, 'cruise_ship': 94,
            'cup': 52, 'diamond': 109, 'dishwasher': 47, 'diving_board': 12, 'dog': 225, 'dolphin': 165, 'donut': 65, 'door': 49, 'dragon': 30, 'dresser': 23,
            'drill': 44, 'drums': 18, 'duck': 51, 'dumbbell': 86, 'ear': 58, 'elbow': 74, 'elephant': 188, 'envelope': 60, 'eraser': 34, 'eye': 168,
            'eyeglasses': 118, 'face': 110, 'fan': 49, 'feather': 432, 'fence': 99, 'finger': 71, 'fire_hydrant': 59, 'fireplace': 98, 'firetruck': 39, 'fish': 195,
            'flamingo': 39, 'flashlight': 62, 'flip_flops': 53, 'floor_lamp': 100, 'flower': 140, 'flying_saucer': 40, 'foot': 111, 'fork': 63, 'frog': 118, 'frying_pan': 68,
            'garden': 291, 'garden_hose': 48, 'giraffe': 172, 'goatee': 236, 'golf_club': 169, 'grapes': 171, 'grass': 312, 'guitar': 204, 'hamburger': 210, 'hammer': 70,
            'hand': 268, 'harp': 37, 'hat': 201, 'headphones': 224, 'hedgehog': 48, 'helicopter': 216, 'helmet': 263, 'hexagon': 362, 'hockey_puck': 59, 'hockey_stick': 197,
            'horse': 216, 'hospital': 48, 'hot_air_balloon': 48, 'hot_dog': 138, 'hot_tub': 86, 'hourglass': 100, 'house': 306, 'house_plant': 292, 'hurricane': 68, 'ice_cream': 187,
            'jacket': 82, 'jail': 26, 'kangaroo': 60, 'key': 68, 'keyboard': 95, 'knee': 56, 'knife': 108, 'ladder': 96, 'lantern': 58, 'laptop': 118,
            'leaf': 96, 'leg': 174, 'light_bulb': 185, 'lighter': 27, 'lighthouse': 66, 'lightning': 68, 'line': 210, 'lion': 64, 'lipstick': 104, 'lobster': 47,
            'lollipop': 28, 'mailbox': 45, 'map': 206, 'marker': 44, 'matches': 19, 'megaphone': 91, 'mermaid': 30, 'microphone': 70, 'microwave': 114, 'monkey': 85,
            'moon': 195, 'mosquito': 232, 'motorbike': 209, 'mountain': 57, 'mouse': 50, 'moustache': 28, 'mouth': 103, 'mug': 41, 'mushroom': 298, 'nail': 256,
            'necklace': 115, 'nose': 226, 'ocean': 47, 'octagon': 115, 'octopus': 49, 'onion': 62, 'oven': 59, 'owl': 114, 'paint_can': 42, 'paintbrush': 28,
            'palm_tree': 277, 'panda': 86, 'pants': 173, 'paper_clip': 25, 'parachute': 60, 'parrot': 62, 'passport': 120, 'peanut': 60, 'pear': 115, 'peas': 120,
            'pencil': 369, 'penguin': 201, 'piano': 66, 'pickup_truck': 116, 'picture_frame': 60, 'pig': 203, 'pillow': 170, 'pineapple': 139, 'pizza': 157, 'pliers': 65,
            'police_car': 51, 'pond': 72, 'pool': 173, 'popsicle': 79, 'postcard': 37, 'potato': 61, 'power_outlet': 76, 'purse': 119, 'rabbit': 135, 'raccoon': 24,
            'radio': 101, 'rain': 78, 'rainbow': 44, 'rake': 66, 'remote_control': 70, 'rhinoceros': 91, 'rifle': 149, 'river': 155, 'roller_coaster': 46, 'rollerskates': 49,
            'sailboat': 119, 'sandwich': 110, 'saw': 34, 'saxophone': 74, 'school_bus': 142, 'scissors': 103, 'scorpion': 53, 'screwdriver': 34, 'sea_turtle': 190, 'see_saw': 28,
            'shark': 279, 'sheep': 70, 'shoe': 291, 'shorts': 29, 'shovel': 17, 'sink': 32, 'skateboard': 50, 'skull': 29, 'skyscraper': 159, 'sleeping_bag': 14,
            'smiley_face': 46, 'snail': 18, 'snake': 57, 'snorkel': 81, 'snowflake': 41, 'snowman': 123, 'soccer_ball': 163, 'sock': 453, 'speedboat': 76, 'spider': 154,
            'spoon': 127, 'spreadsheet': 397, 'square': 211, 'squiggle': 115, 'squirrel': 180, 'stairs': 282, 'star': 204, 'steak': 360, 'stereo': 334, 'stethoscope': 107,
            'stitches': 285, 'stop_sign': 54, 'stove': 255, 'strawberry': 308, 'streetlight': 113, 'string_bean': 87, 'submarine': 183, 'suitcase': 224, 'sun': 352, 'swan': 52,
            'sweater': 92, 'swing_set': 35, 'sword': 124, 'syringe': 240, 't-shirt': 320, 'table': 736, 'teapot': 209, 'teddy-bear': 407, 'telephone': 279, 'television': 546,
            'tennis_racquet': 195, 'tent': 234, 'tiger': 285, 'toaster': 337, 'toe': 407, 'toilet': 519, 'tooth': 473, 'toothbrush': 556, 'toothpaste': 468, 'tornado': 329,
            'tractor': 316, 'traffic_light': 280, 'train': 373, 'tree': 511, 'triangle': 364, 'trombone': 195, 'truck': 678, 'trumpet': 247, 'umbrella': 511, 'underwear': 354,
            'van': 806, 'vase': 319, 'violin': 282, 'washing_machine': 519, 'watermelon': 401, 'waterslide': 328, 'whale': 432, 'wheel': 385, 'windmill': 372, 'wine_bottle': 442,
            'wine_glass': 628, 'wristwatch': 470, 'yoga': 447, 'zebra': 306, 'zigzag': 412,
        },
        'painting': {
            'The_Eiffel_Tower': 321, 'The_Great_Wall_of_China': 159, 'The_Mona_Lisa': 191, 'aircraft_carrier': 133, 'airplane': 212, 'alarm_clock': 84, 'ambulance': 74, 'angel': 504, 'animal_migration': 604, 'ant': 235,
            'anvil': 152, 'apple': 445, 'arm': 422, 'asparagus': 408, 'axe': 219, 'backpack': 265, 'banana': 359, 'bandage': 197, 'barn': 426, 'baseball': 122,
            'baseball_bat': 145, 'basket': 417, 'basketball': 276, 'bat': 306, 'bathtub': 45, 'beach': 499, 'bear': 379, 'beard': 373, 'bed': 46, 'bee': 313,
            'belt': 17, 'bench': 167, 'bicycle': 196, 'binoculars': 148, 'bird': 222, 'birthday_cake': 119, 'blackberry': 14, 'blueberry': 167, 'book': 65, 'boomerang': 41,
            'bottlecap': 538, 'bowtie': 292, 'bracelet': 150, 'brain': 233, 'bread': 315, 'bridge': 471, 'broccoli': 100, 'broom': 84, 'bucket': 61, 'bulldozer': 58,
            'bus': 112, 'bush': 67, 'butterfly': 387, 'cactus': 122, 'cake': 172, 'calculator': 12, 'calendar': 44, 'camel': 289, 'camera': 156, 'camouflage': 72,
            'campfire': 217, 'candle': 261, 'cannon': 54, 'canoe': 395, 'car': 45, 'carrot': 265, 'castle': 225, 'cat': 344, 'ceiling_fan': 38, 'cell_phone': 136,
            'cello': 158, 'chair': 53, 'chandelier': 57, 'church': 142, 'circle': 292, 'clarinet': 89, 'clock': 266, 'cloud': 278, 'coffee_cup': 185, 'compass': 78,
            'computer': 19, 'cookie': 54, 'cooler': 13, 'couch': 26, 'cow': 156, 'crab': 153, 'crayon': 19, 'crocodile': 120, 'crown': 81, 'cruise_ship': 223,
            'cup': 582, 'diamond': 17, 'dishwasher': 107, 'diving_board': 127, 'dog': 721, 'dolphin': 401, 'donut': 373, 'door': 347, 'dragon': 231, 'dresser': 141,
            'drill': 21, 'drums': 205, 'duck': 419, 'dumbbell': 189, 'ear': 187, 'elbow': 97, 'elephant': 425, 'envelope': 291, 'eraser': 17, 'eye': 292,
            'eyeglasses': 83, 'face': 20, 'fan': 16, 'feather': 344, 'fence': 49, 'finger': 57, 'fire_hydrant': 29, 'fireplace': 15, 'firetruck': 359, 'fish': 429,
            'flamingo': 224, 'flashlight': 418, 'flip_flops': 206, 'floor_lamp': 10, 'flower': 485, 'flying_saucer': 242, 'foot': 86, 'fork': 84, 'frog': 167, 'frying_pan': 169,
            'garden': 213, 'garden_hose': 179, 'giraffe': 105, 'goatee': 129, 'golf_club': 650, 'grapes': 318, 'grass': 332, 'guitar': 203, 'hamburger': 147, 'hammer': 46,
            'hand': 262, 'harp': 224, 'hat': 400, 'headphones': 181, 'hedgehog': 248, 'helicopter': 257, 'helmet': 27, 'hexagon': 160, 'hockey_puck': 236, 'hockey_stick': 194,
            'horse': 521, 'hospital': 50, 'hot_air_balloon': 453, 'hot_dog': 148, 'hot_tub': 197, 'hourglass': 206, 'house': 105, 'house_plant': 416, 'hurricane': 133, 'ice_cream': 311,
            'jacket': 272, 'jail': 54, 'kangaroo': 214, 'key': 97, 'keyboard': 370, 'knee': 130, 'knife': 30, 'ladder': 418, 'lantern': 218, 'laptop': 161,
            'leaf': 504, 'leg': 178, 'light_bulb': 482, 'lighter': 27, 'lighthouse': 411, 'lightning': 199, 'line': 502, 'lion': 505, 'lipstick': 196, 'lobster': 254,
            'lollipop': 252, 'mailbox': 101, 'map': 423, 'marker': 103, 'matches': 56, 'megaphone': 160, 'mermaid': 99, 'microphone': 152, 'microwave': 10, 'monkey': 405,
            'moon': 324, 'mosquito': 65, 'motorbike': 106, 'mountain': 319, 'mouse': 445, 'moustache': 430, 'mouth': 51, 'mug': 500, 'mushroom': 254, 'nail': 838,
            'necklace': 347, 'nose': 512, 'ocean': 475, 'octagon': 19, 'octopus': 331, 'onion': 471, 'oven': 11, 'owl': 496, 'paint_can': 172, 'paintbrush': 365,
            'palm_tree': 607, 'panda': 264, 'pants': 381, 'paper_clip': 112, 'parachute': 140, 'parrot': 336, 'passport': 34, 'peanut': 82, 'pear': 448, 'peas': 90,
            'pencil': 183, 'penguin': 447, 'piano': 296, 'pickup_truck': 143, 'picture_frame': 372, 'pig': 326, 'pillow': 144, 'pineapple': 333, 'pizza': 127, 'pliers': 293,
            'police_car': 87, 'pond': 603, 'pool': 90, 'popsicle': 171, 'postcard': 88, 'potato': 58, 'power_outlet': 102, 'purse': 49, 'rabbit': 269, 'raccoon': 249,
            'radio': 65, 'rain': 352, 'rainbow': 84, 'rake': 58, 'remote_control': 111, 'rhinoceros': 220, 'rifle': 240, 'river': 558, 'roller_coaster': 75, 'rollerskates': 322,
            'sailboat': 322, 'sandwich': 139, 'saw': 150, 'saxophone': 358, 'school_bus': 66, 'scissors': 65, 'scorpion': 133, 'screwdriver': 73, 'sea_turtle': 410, 'see_saw': 166,
            'shark': 269, 'sheep': 334, 'shoe': 260, 'shorts': 161, 'shovel': 112, 'sink': 94, 'skateboard': 152, 'skull': 189, 'skyscraper': 179, 'sleeping_bag': 17,
            'smiley_face': 77, 'snail': 321, 'snake': 425, 'snorkel': 179, 'snowflake': 405, 'snowman': 901, 'soccer_ball': 268, 'sock': 31, 'speedboat': 141, 'spider': 308,
            'spoon': 158, 'spreadsheet': 34, 'square': 144, 'squiggle': 674, 'squirrel': 779, 'stairs': 27, 'star': 98, 'steak': 50, 'stereo': 12, 'stethoscope': 346,
            'stitches': 17, 'stop_sign': 87, 'stove': 16, 'strawberry': 530, 'streetlight': 537, 'string_bean': 70, 'submarine': 550, 'suitcase': 187, 'sun': 572, 'swan': 507,
            'sweater': 153, 'swing_set': 129, 'sword': 470, 'syringe': 10, 't-shirt': 12, 'table': 104, 'teapot': 391, 'teddy-bear': 301, 'telephone': 78, 'television': 51,
            'tennis_racquet': 18, 'tent': 141, 'tiger': 422, 'toaster': 107, 'toe': 12, 'toilet': 31, 'tooth': 109, 'toothbrush': 11, 'toothpaste': 31, 'tornado': 373,
            'tractor': 183, 'traffic_light': 60, 'train': 406, 'tree': 571, 'triangle': 298, 'trombone': 175, 'truck': 158, 'trumpet': 122, 'umbrella': 299, 'underwear': 12,
            'van': 12, 'vase': 262, 'violin': 203, 'washing_machine': 15, 'watermelon': 410, 'waterslide': 12, 'whale': 357, 'wheel': 19, 'windmill': 397, 'wine_bottle': 59,
            'wine_glass': 168, 'wristwatch': 18, 'yoga': 161, 'zebra': 298, 'zigzag': 110,
        },
        'quickdraw': {
            'The_Eiffel_Tower': 500, 'The_Great_Wall_of_China': 500, 'The_Mona_Lisa': 500, 'aircraft_carrier': 500, 'airplane': 500, 'alarm_clock': 500, 'ambulance': 500, 'angel': 500, 'animal_migration': 500, 'ant': 500,
            'anvil': 500, 'apple': 500, 'arm': 500, 'asparagus': 500, 'axe': 500, 'backpack': 500, 'banana': 500, 'bandage': 500, 'barn': 500, 'baseball': 500,
            'baseball_bat': 500, 'basket': 500, 'basketball': 500, 'bat': 500, 'bathtub': 500, 'beach': 500, 'bear': 500, 'beard': 500, 'bed': 500, 'bee': 500,
            'belt': 500, 'bench': 500, 'bicycle': 500, 'binoculars': 500, 'bird': 500, 'birthday_cake': 500, 'blackberry': 500, 'blueberry': 500, 'book': 500, 'boomerang': 500,
            'bottlecap': 500, 'bowtie': 500, 'bracelet': 500, 'brain': 500, 'bread': 500, 'bridge': 500, 'broccoli': 500, 'broom': 500, 'bucket': 500, 'bulldozer': 500,
            'bus': 500, 'bush': 500, 'butterfly': 500, 'cactus': 500, 'cake': 500, 'calculator': 500, 'calendar': 500, 'camel': 500, 'camera': 500, 'camouflage': 500,
            'campfire': 500, 'candle': 500, 'cannon': 500, 'canoe': 500, 'car': 500, 'carrot': 500, 'castle': 500, 'cat': 500, 'ceiling_fan': 500, 'cell_phone': 500,
            'cello': 500, 'chair': 500, 'chandelier': 500, 'church': 500, 'circle': 500, 'clarinet': 500, 'clock': 500, 'cloud': 500, 'coffee_cup': 500, 'compass': 500,
            'computer': 500, 'cookie': 500, 'cooler': 500, 'couch': 500, 'cow': 500, 'crab': 500, 'crayon': 500, 'crocodile': 500, 'crown': 500, 'cruise_ship': 500,
            'cup': 500, 'diamond': 500, 'dishwasher': 500, 'diving_board': 500, 'dog': 500, 'dolphin': 500, 'donut': 500, 'door': 500, 'dragon': 500, 'dresser': 500,
            'drill': 500, 'drums': 500, 'duck': 500, 'dumbbell': 500, 'ear': 500, 'elbow': 500, 'elephant': 500, 'envelope': 500, 'eraser': 500, 'eye': 500,
            'eyeglasses': 500, 'face': 500, 'fan': 500, 'feather': 500, 'fence': 500, 'finger': 500, 'fire_hydrant': 500, 'fireplace': 500, 'firetruck': 500, 'fish': 500,
            'flamingo': 500, 'flashlight': 500, 'flip_flops': 500, 'floor_lamp': 500, 'flower': 500, 'flying_saucer': 500, 'foot': 500, 'fork': 500, 'frog': 500, 'frying_pan': 500,
            'garden': 500, 'garden_hose': 500, 'giraffe': 500, 'goatee': 500, 'golf_club': 500, 'grapes': 500, 'grass': 500, 'guitar': 500, 'hamburger': 500, 'hammer': 500,
            'hand': 500, 'harp': 500, 'hat': 500, 'headphones': 500, 'hedgehog': 500, 'helicopter': 500, 'helmet': 500, 'hexagon': 500, 'hockey_puck': 500, 'hockey_stick': 500,
            'horse': 500, 'hospital': 500, 'hot_air_balloon': 500, 'hot_dog': 500, 'hot_tub': 500, 'hourglass': 500, 'house': 500, 'house_plant': 500, 'hurricane': 500, 'ice_cream': 500,
            'jacket': 500, 'jail': 500, 'kangaroo': 500, 'key': 500, 'keyboard': 500, 'knee': 500, 'knife': 500, 'ladder': 500, 'lantern': 500, 'laptop': 500,
            'leaf': 500, 'leg': 500, 'light_bulb': 500, 'lighter': 500, 'lighthouse': 500, 'lightning': 500, 'line': 500, 'lion': 500, 'lipstick': 500, 'lobster': 500,
            'lollipop': 500, 'mailbox': 500, 'map': 500, 'marker': 500, 'matches': 500, 'megaphone': 500, 'mermaid': 500, 'microphone': 500, 'microwave': 500, 'monkey': 500,
            'moon': 500, 'mosquito': 500, 'motorbike': 500, 'mountain': 500, 'mouse': 500, 'moustache': 500, 'mouth': 500, 'mug': 500, 'mushroom': 500, 'nail': 500,
            'necklace': 500, 'nose': 500, 'ocean': 500, 'octagon': 500, 'octopus': 500, 'onion': 500, 'oven': 500, 'owl': 500, 'paint_can': 500, 'paintbrush': 500,
            'palm_tree': 500, 'panda': 500, 'pants': 500, 'paper_clip': 500, 'parachute': 500, 'parrot': 500, 'passport': 500, 'peanut': 500, 'pear': 500, 'peas': 500,
            'pencil': 500, 'penguin': 500, 'piano': 500, 'pickup_truck': 500, 'picture_frame': 500, 'pig': 500, 'pillow': 500, 'pineapple': 500, 'pizza': 500, 'pliers': 500,
            'police_car': 500, 'pond': 500, 'pool': 500, 'popsicle': 500, 'postcard': 500, 'potato': 500, 'power_outlet': 500, 'purse': 500, 'rabbit': 500, 'raccoon': 500,
            'radio': 500, 'rain': 500, 'rainbow': 500, 'rake': 500, 'remote_control': 500, 'rhinoceros': 500, 'rifle': 500, 'river': 500, 'roller_coaster': 500, 'rollerskates': 500,
            'sailboat': 500, 'sandwich': 500, 'saw': 500, 'saxophone': 500, 'school_bus': 500, 'scissors': 500, 'scorpion': 500, 'screwdriver': 500, 'sea_turtle': 500, 'see_saw': 500,
            'shark': 500, 'sheep': 500, 'shoe': 500, 'shorts': 500, 'shovel': 500, 'sink': 500, 'skateboard': 500, 'skull': 500, 'skyscraper': 500, 'sleeping_bag': 500,
            'smiley_face': 500, 'snail': 500, 'snake': 500, 'snorkel': 500, 'snowflake': 500, 'snowman': 500, 'soccer_ball': 500, 'sock': 500, 'speedboat': 500, 'spider': 500,
            'spoon': 500, 'spreadsheet': 500, 'square': 500, 'squiggle': 500, 'squirrel': 500, 'stairs': 500, 'star': 500, 'steak': 500, 'stereo': 500, 'stethoscope': 500,
            'stitches': 500, 'stop_sign': 500, 'stove': 500, 'strawberry': 500, 'streetlight': 500, 'string_bean': 500, 'submarine': 500, 'suitcase': 500, 'sun': 500, 'swan': 500,
            'sweater': 500, 'swing_set': 500, 'sword': 500, 'syringe': 500, 't-shirt': 500, 'table': 500, 'teapot': 500, 'teddy-bear': 500, 'telephone': 500, 'television': 500,
            'tennis_racquet': 500, 'tent': 500, 'tiger': 500, 'toaster': 500, 'toe': 500, 'toilet': 500, 'tooth': 500, 'toothbrush': 500, 'toothpaste': 500, 'tornado': 500,
            'tractor': 500, 'traffic_light': 500, 'train': 500, 'tree': 500, 'triangle': 500, 'trombone': 500, 'truck': 500, 'trumpet': 500, 'umbrella': 500, 'underwear': 500,
            'van': 500, 'vase': 500, 'violin': 500, 'washing_machine': 500, 'watermelon': 500, 'waterslide': 500, 'whale': 500, 'wheel': 500, 'windmill': 500, 'wine_bottle': 500,
            'wine_glass': 500, 'wristwatch': 500, 'yoga': 500, 'zebra': 500, 'zigzag': 500,
        },
        'real': {
            'The_Eiffel_Tower': 553, 'The_Great_Wall_of_China': 530, 'The_Mona_Lisa': 289, 'aircraft_carrier': 390, 'airplane': 218, 'alarm_clock': 521, 'ambulance': 623, 'angel': 31, 'animal_migration': 444, 'ant': 381,
            'anvil': 332, 'apple': 54, 'arm': 235, 'asparagus': 659, 'axe': 382, 'backpack': 439, 'banana': 258, 'bandage': 399, 'barn': 313, 'baseball': 87,
            'baseball_bat': 118, 'basket': 444, 'basketball': 237, 'bat': 361, 'bathtub': 517, 'beach': 622, 'bear': 585, 'beard': 728, 'bed': 724, 'bee': 452,
            'belt': 661, 'bench': 662, 'bicycle': 705, 'binoculars': 402, 'bird': 803, 'birthday_cake': 307, 'blackberry': 568, 'blueberry': 733, 'book': 731, 'boomerang': 628,
            'bottlecap': 606, 'bowtie': 533, 'bracelet': 715, 'brain': 724, 'bread': 794, 'bridge': 769, 'broccoli': 679, 'broom': 639, 'bucket': 335, 'bulldozer': 635,
            'bus': 745, 'bush': 31, 'butterfly': 658, 'cactus': 658, 'cake': 786, 'calculator': 374, 'calendar': 176, 'camel': 493, 'camera': 480, 'camouflage': 124,
            'campfire': 489, 'candle': 621, 'cannon': 300, 'canoe': 703, 'car': 564, 'carrot': 565, 'castle': 682, 'cat': 796, 'ceiling_fan': 217, 'cell_phone': 520,
            'cello': 450, 'chair': 320, 'chandelier': 393, 'church': 668, 'circle': 259, 'clarinet': 407, 'clock': 619, 'cloud': 324, 'coffee_cup': 643, 'compass': 272,
            'computer': 362, 'cookie': 677, 'cooler': 528, 'couch': 601, 'cow': 541, 'crab': 717, 'crayon': 512, 'crocodile': 713, 'crown': 170, 'cruise_ship': 632,
            'cup': 406, 'diamond': 577, 'dishwasher': 508, 'diving_board': 593, 'dog': 782, 'dolphin': 581, 'donut': 630, 'door': 371, 'dragon': 485, 'dresser': 234,
            'drill': 573, 'drums': 769, 'duck': 404, 'dumbbell': 581, 'ear': 348, 'elbow': 398, 'elephant': 789, 'envelope': 665, 'eraser': 356, 'eye': 695,
            'eyeglasses': 680, 'face': 696, 'fan': 460, 'feather': 505, 'fence': 770, 'finger': 625, 'fire_hydrant': 579, 'fireplace': 700, 'firetruck': 562, 'fish': 479,
            'flamingo': 542, 'flashlight': 461, 'flip_flops': 525, 'floor_lamp': 246, 'flower': 360, 'flying_saucer': 396, 'foot': 558, 'fork': 351, 'frog': 761, 'frying_pan': 399,
            'garden': 815, 'garden_hose': 534, 'giraffe': 594, 'goatee': 562, 'golf_club': 552, 'grapes': 734, 'grass': 378, 'guitar': 632, 'hamburger': 559, 'hammer': 347,
            'hand': 563, 'harp': 649, 'hat': 529, 'headphones': 551, 'hedgehog': 727, 'helicopter': 804, 'helmet': 622, 'hexagon': 592, 'hockey_puck': 400, 'hockey_stick': 394,
            'horse': 645, 'hospital': 674, 'hot_air_balloon': 732, 'hot_dog': 644, 'hot_tub': 757, 'hourglass': 289, 'house': 374, 'house_plant': 484, 'hurricane': 420, 'ice_cream': 657,
            'jacket': 457, 'jail': 587, 'kangaroo': 613, 'key': 229, 'keyboard': 503, 'knee': 505, 'knife': 582, 'ladder': 442, 'lantern': 526, 'laptop': 387,
            'leaf': 414, 'leg': 659, 'light_bulb': 262, 'lighter': 587, 'lighthouse': 722, 'lightning': 560, 'line': 102, 'lion': 516, 'lipstick': 446, 'lobster': 649,
            'lollipop': 607, 'mailbox': 595, 'map': 507, 'marker': 336, 'matches': 333, 'megaphone': 560, 'mermaid': 449, 'microphone': 562, 'microwave': 338, 'monkey': 699,
            'moon': 568, 'mosquito': 562, 'motorbike': 772, 'mountain': 791, 'mouse': 147, 'moustache': 424, 'mouth': 457, 'mug': 598, 'mushroom': 788, 'nail': 674,
            'necklace': 697, 'nose': 362, 'ocean': 591, 'octagon': 465, 'octopus': 726, 'onion': 599, 'oven': 492, 'owl': 757, 'paint_can': 560, 'paintbrush': 413,
            'palm_tree': 333, 'panda': 587, 'pants': 398, 'paper_clip': 549, 'parachute': 629, 'parrot': 781, 'passport': 535, 'peanut': 423, 'pear': 438, 'peas': 680,
            'pencil': 461, 'penguin': 700, 'piano': 570, 'pickup_truck': 619, 'picture_frame': 207, 'pig': 577, 'pillow': 656, 'pineapple': 673, 'pizza': 600, 'pliers': 453,
            'police_car': 740, 'pond': 777, 'pool': 680, 'popsicle': 639, 'postcard': 636, 'potato': 608, 'power_outlet': 620, 'purse': 544, 'rabbit': 695, 'raccoon': 676,
            'radio': 398, 'rain': 274, 'rainbow': 231, 'rake': 594, 'remote_control': 554, 'rhinoceros': 684, 'rifle': 520, 'river': 651, 'roller_coaster': 637, 'rollerskates': 493,
            'sailboat': 422, 'sandwich': 579, 'saw': 118, 'saxophone': 482, 'school_bus': 478, 'scissors': 568, 'scorpion': 447, 'screwdriver': 417, 'sea_turtle': 621, 'see_saw': 273,
            'shark': 183, 'sheep': 796, 'shoe': 587, 'shorts': 443, 'shovel': 450, 'sink': 231, 'skateboard': 557, 'skull': 329, 'skyscraper': 284, 'sleeping_bag': 406,
            'smiley_face': 226, 'snail': 465, 'snake': 501, 'snorkel': 689, 'snowflake': 66, 'snowman': 114, 'soccer_ball': 272, 'sock': 531, 'speedboat': 620, 'spider': 593,
            'spoon': 534, 'spreadsheet': 751, 'square': 98, 'squiggle': 71, 'squirrel': 693, 'stairs': 353, 'star': 61, 'steak': 758, 'stereo': 211, 'stethoscope': 496,
            'stitches': 207, 'stop_sign': 168, 'stove': 614, 'strawberry': 454, 'streetlight': 463, 'string_bean': 491, 'submarine': 607, 'suitcase': 432, 'sun': 161, 'swan': 326,
            'sweater': 579, 'swing_set': 556, 'sword': 591, 'syringe': 589, 't-shirt': 367, 'table': 563, 'teapot': 631, 'teddy-bear': 528, 'telephone': 479, 'television': 400,
            'tennis_racquet': 456, 'tent': 590, 'tiger': 607, 'toaster': 536, 'toe': 356, 'toilet': 583, 'tooth': 257, 'toothbrush': 582, 'toothpaste': 511, 'tornado': 497,
            'tractor': 636, 'traffic_light': 379, 'train': 681, 'tree': 536, 'triangle': 376, 'trombone': 484, 'truck': 673, 'trumpet': 391, 'umbrella': 362, 'underwear': 286,
            'van': 442, 'vase': 632, 'violin': 512, 'washing_machine': 466, 'watermelon': 671, 'waterslide': 606, 'whale': 671, 'wheel': 410, 'windmill': 635, 'wine_bottle': 407,
            'wine_glass': 338, 'wristwatch': 553, 'yoga': 371, 'zebra': 683, 'zigzag': 515,
        },
        'sketch': {
            'The_Eiffel_Tower': 276, 'The_Great_Wall_of_China': 148, 'The_Mona_Lisa': 145, 'aircraft_carrier': 63, 'airplane': 331, 'alarm_clock': 202, 'ambulance': 115, 'angel': 299, 'animal_migration': 112, 'ant': 111,
            'anvil': 91, 'apple': 181, 'arm': 249, 'asparagus': 209, 'axe': 219, 'backpack': 220, 'banana': 204, 'bandage': 56, 'barn': 201, 'baseball': 48,
            'baseball_bat': 196, 'basket': 192, 'basketball': 160, 'bat': 160, 'bathtub': 210, 'beach': 79, 'bear': 178, 'beard': 481, 'bed': 188, 'bee': 144,
            'belt': 125, 'bench': 290, 'bicycle': 343, 'binoculars': 266, 'bird': 306, 'birthday_cake': 233, 'blackberry': 60, 'blueberry': 129, 'book': 146, 'boomerang': 120,
            'bottlecap': 139, 'bowtie': 327, 'bracelet': 300, 'brain': 270, 'bread': 276, 'bridge': 335, 'broccoli': 181, 'broom': 234, 'bucket': 162, 'bulldozer': 199,
            'bus': 233, 'bush': 626, 'butterfly': 249, 'cactus': 61, 'cake': 77, 'calculator': 69, 'calendar': 59, 'camel': 130, 'camera': 109, 'camouflage': 69,
            'campfire': 86, 'candle': 77, 'cannon': 93, 'canoe': 129, 'car': 145, 'carrot': 34, 'castle': 56, 'cat': 130, 'ceiling_fan': 25, 'cell_phone': 23,
            'cello': 64, 'chair': 96, 'chandelier': 34, 'church': 35, 'circle': 202, 'clarinet': 33, 'clock': 44, 'cloud': 100, 'coffee_cup': 33, 'compass': 14,
            'computer': 31, 'cookie': 33, 'cooler': 90, 'couch': 60, 'cow': 17, 'crab': 152, 'crayon': 285, 'crocodile': 161, 'crown': 176, 'cruise_ship': 158,
            'cup': 396, 'diamond': 117, 'dishwasher': 40, 'diving_board': 71, 'dog': 311, 'dolphin': 85, 'donut': 127, 'door': 361, 'dragon': 196, 'dresser': 13,
            'drill': 144, 'drums': 214, 'duck': 276, 'dumbbell': 190, 'ear': 199, 'elbow': 155, 'elephant': 266, 'envelope': 183, 'eraser': 51, 'eye': 489,
            'eyeglasses': 219, 'face': 452, 'fan': 66, 'feather': 336, 'fence': 140, 'finger': 283, 'fire_hydrant': 148, 'fireplace': 123, 'firetruck': 328, 'fish': 373,
            'flamingo': 142, 'flashlight': 95, 'flip_flops': 120, 'floor_lamp': 278, 'flower': 336, 'flying_saucer': 137, 'foot': 261, 'fork': 176, 'frog': 203, 'frying_pan': 132,
            'garden': 98, 'garden_hose': 84, 'giraffe': 186, 'goatee': 219, 'golf_club': 695, 'grapes': 287, 'grass': 173, 'guitar': 183, 'hamburger': 185, 'hammer': 71,
            'hand': 264, 'harp': 45, 'hat': 77, 'headphones': 188, 'hedgehog': 109, 'helicopter': 200, 'helmet': 210, 'hexagon': 116, 'hockey_puck': 95, 'hockey_stick': 119,
            'horse': 103, 'hospital': 24, 'hot_air_balloon': 170, 'hot_dog': 143, 'hot_tub': 49, 'hourglass': 134, 'house': 144, 'house_plant': 156, 'hurricane': 99, 'ice_cream': 184,
            'jacket': 84, 'jail': 94, 'kangaroo': 122, 'key': 137, 'keyboard': 64, 'knee': 273, 'knife': 129, 'ladder': 244, 'lantern': 40, 'laptop': 319,
            'leaf': 378, 'leg': 145, 'light_bulb': 405, 'lighter': 118, 'lighthouse': 384, 'lightning': 94, 'line': 25, 'lion': 330, 'lipstick': 110, 'lobster': 174,
            'lollipop': 106, 'mailbox': 151, 'map': 193, 'marker': 240, 'matches': 56, 'megaphone': 189, 'mermaid': 228, 'microphone': 156, 'microwave': 170, 'monkey': 166,
            'moon': 155, 'mosquito': 144, 'motorbike': 209, 'mountain': 195, 'mouse': 127, 'moustache': 107, 'mouth': 172, 'mug': 186, 'mushroom': 252, 'nail': 23,
            'necklace': 114, 'nose': 103, 'ocean': 77, 'octagon': 117, 'octopus': 149, 'onion': 158, 'oven': 176, 'owl': 202, 'paint_can': 34, 'paintbrush': 75,
            'palm_tree': 166, 'panda': 79, 'pants': 136, 'paper_clip': 119, 'parachute': 233, 'parrot': 266, 'passport': 97, 'peanut': 130, 'pear': 183, 'peas': 81,
            'pencil': 26, 'penguin': 209, 'piano': 119, 'pickup_truck': 188, 'picture_frame': 115, 'pig': 227, 'pillow': 115, 'pineapple': 131, 'pizza': 202, 'pliers': 163,
            'police_car': 119, 'pond': 95, 'pool': 103, 'popsicle': 117, 'postcard': 49, 'potato': 83, 'power_outlet': 95, 'purse': 228, 'rabbit': 94, 'raccoon': 348,
            'radio': 165, 'rain': 235, 'rainbow': 46, 'rake': 93, 'remote_control': 47, 'rhinoceros': 183, 'rifle': 122, 'river': 111, 'roller_coaster': 61, 'rollerskates': 141,
            'sailboat': 361, 'sandwich': 132, 'saw': 110, 'saxophone': 310, 'school_bus': 405, 'scissors': 437, 'scorpion': 455, 'screwdriver': 373, 'sea_turtle': 254, 'see_saw': 519,
            'shark': 532, 'sheep': 475, 'shoe': 645, 'shorts': 529, 'shovel': 630, 'sink': 464, 'skateboard': 419, 'skull': 600, 'skyscraper': 466, 'sleeping_bag': 591,
            'smiley_face': 441, 'snail': 405, 'snake': 470, 'snorkel': 397, 'snowflake': 460, 'snowman': 712, 'soccer_ball': 377, 'sock': 453, 'speedboat': 487, 'spider': 645,
            'spoon': 406, 'spreadsheet': 677, 'square': 727, 'squiggle': 442, 'squirrel': 389, 'stairs': 525, 'star': 205, 'steak': 238, 'stereo': 90, 'stethoscope': 237,
            'stitches': 34, 'stop_sign': 109, 'stove': 269, 'strawberry': 198, 'streetlight': 268, 'string_bean': 68, 'submarine': 207, 'suitcase': 309, 'sun': 258, 'swan': 236,
            'sweater': 167, 'swing_set': 96, 'sword': 384, 'syringe': 222, 't-shirt': 155, 'table': 300, 'teapot': 327, 'teddy-bear': 238, 'telephone': 255, 'television': 127,
            'tennis_racquet': 202, 'tent': 339, 'tiger': 386, 'toaster': 267, 'toe': 78, 'toilet': 118, 'tooth': 181, 'toothbrush': 235, 'toothpaste': 198, 'tornado': 211,
            'tractor': 263, 'traffic_light': 127, 'train': 240, 'tree': 555, 'triangle': 303, 'trombone': 191, 'truck': 265, 'trumpet': 188, 'umbrella': 297, 'underwear': 132,
            'van': 138, 'vase': 187, 'violin': 203, 'washing_machine': 155, 'watermelon': 128, 'waterslide': 115, 'whale': 272, 'wheel': 166, 'windmill': 245, 'wine_bottle': 274,
            'wine_glass': 245, 'wristwatch': 224, 'yoga': 251, 'zebra': 278, 'zigzag': 144,
        },
    },
}

NCOLS = {
    DATASETS['pacs']: 4,
    DATASETS['vlcs']: 5,
    DATASETS['officehome']: 10,
    DATASETS['domainnet']: 10,
}

BLANKS = '    '


def dict_print(name, mapping, ncols = 1, offset=0, clear_mark=False, need_upper=False, file=sys.stdout):
    if name is not None:
        file.write('%s = {\n' % name)

    if not isinstance(mapping, dict):
        raise NotImplementedError('>_<')

    def get_parent(key):
        parent = None
        for dataset in DOMAINS.keys():
            for domain in DOMAINS[dataset].keys():
                if domain.lower() == key:
                    parent = dataset
                    break
            if parent is not None:
                break
        return parent
    
    # \n flag
    blanks, length = ''.join([BLANKS] * (offset + 1)), len(mapping)
    flags, mark = [], '"' if clear_mark == False else ''
    for ind, val in enumerate(mapping.values()):
        flags.append(True if (ind + 1) % ncols == 0 else False)
        if isinstance(val, dict):
            if len(flags) > 1:
                flags[-2] = True
            flags[-1] = True
    flags[-1] = True

    # print 
    for ind, (k, v) in enumerate(mapping.items()):
        file.write(f'{blanks}' if flags[ind - 1] else '')
        file.write(f'{mark}{k.upper() if need_upper else k}{mark}: ')
        if not isinstance(v, dict):
            file.write(f'{mark}{v}{mark}' if isinstance(v, str) else f'{v}')
        else:
            offset += 1
            file.write('{\n')
            if len(v) > 0:
                parent = get_parent(name)
                dict_print(None, v, NCOLS[parent] if parent is not None else (length + 1) // 2, offset=offset, file=file)
            file.write('%s}' % blanks)
            offset -= 1
        file.write(f',')
        file.write('\n' if flags[ind] else ' ')

    if name is not None:
        file.write('}\n')


def domain_print(domains, file=sys.stdout):
    for k, v in domains.items():
        file.write('%s = "%s"\n' % (k.upper(), v))
    file.write('\n')


def main(dataset, valid, many, few, gen, file=sys.stdout):
    domain_print(domains=DOMAINS[DATASETS[dataset]], file=file)

    stats = gen.generate(list(DOMAINS[DATASETS[dataset]].keys()), STATS[DATASETS[dataset]])
    for k, v in stats.items():
        dict_print(name=k.lower(), mapping=v, clear_mark=True, need_upper=True, file=file)

    dict_print(name='mapping', mapping={k.upper(): k.lower() for k in DOMAINS[DATASETS[dataset]].keys()}, clear_mark=True, file=file)
    file.write('\n')
    dict_print(name='thres', mapping={k.upper(): {'many': many, 'few': few} for k in DOMAINS[DATASETS[dataset]].keys()}, clear_mark=True, file=file)
    file.write('\n')
    dict_print(name='validation', mapping={k.upper(): valid for k in DOMAINS[DATASETS[dataset]].keys()}, clear_mark=True, file=file)
    file.write('\n')

    for k, v in stats.items():
        print(k, generate.sorted_items(generate.mean_items(v)))

    return stats


def draw_stats(stats, name=None):
    colors = ['#bdd6fb', '#f5e7af', '#cdeacd', '#f2c6c4', '#dcc0e5', '#fcebde', '#dbeef3', 'd77470']
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    domains, width, lengthwidth, rotate = list(stats.keys()), 0.4, 2, None
    name_classes = [cls.capitalize() for cls in list(list(stats.values())[0].values())[0].keys()]
    start, n_class = lengthwidth - width * (len(domains) - 1) / 2 + width / 2, len(name_classes)

    def print_domain_name(name):
        return '_'.join([item.capitalize() for item in name.split('_')])

    folds = 2 if n_class > 10 else 1
    fig, ax = plt.subplots(folds, len(stats) // folds, sharey=True, figsize=(19.2, 5.5 * folds))
    for ind, target in enumerate(stats.keys()):
        rotate = (90 if n_class > 10 else 0) if rotate is None else rotate
        if n_class > 100:
            left, offset = 1 + ind, len(domains)
        elif n_class > 10:
            left, offset = (1, 2) if ind % 2 == 0 else (2, 2)
        else:
            left, offset = 1, 1
        indices = (ind) if folds == 1 else (ind // (len(stats) // folds), ind % (len(stats) // folds))
        for env, source in enumerate(stats[target].keys()):
            x = lengthwidth * np.arange(len(stats[target][source].values())) + start + width * env
            y = list(stats[target][source].values())
            ax[indices].bar(x, y, width=width, color=colors[domains.index(source)], label=print_domain_name(source))
            for (pos, val) in zip(x, y):
                ax[indices].text(pos, val + (1.25 if val > 50 else 0.25), '%d' % val, ha='center', va='bottom', fontsize=8 if folds == 1 else 10)
        ax[indices].set_xticks(lengthwidth * np.arange(left, n_class + 1, offset))
        if folds == 1 or folds > 1 and (ind // folds) == folds - 1:
            ax[indices].set_xticklabels([name_classes[i] for i in range(left - 1, n_class, offset)], rotation=rotate)
        else:
            ax[indices].set_xticklabels([])
        if folds > 1:
            ax[indices].tick_params(axis='both', which='major', labelsize=18)
        ax[indices].set_xlabel(f'({labels[ind]}) ' + print_domain_name(target), fontsize=12 if folds == 1 else 20)
        if folds > 1:
            ax[indices].legend(fontsize=18)
        else:
            ax[indices].legend()
    plt.tight_layout()
    plt.show() if name is None else plt.savefig(name)
    plt.close()


class Generator(object):
    def __init__(self, valid, test, lower: int = 1):
        assert valid > 0 and test > 0 and lower >= 0

        self.valid = valid
        self.test = test
        self.lower = lower

    @staticmethod
    def sorted(val: dict, func, reserve: bool = True):
        return sorted(val.values(), key=func, reverse=reserve)
    
    @staticmethod
    def sorted_items(val: dict, func, reserve: bool = True):
        return sorted(val.items(), key=func, reverse=reserve)
    
    @staticmethod
    def sorted_items(val: dict, reserve: bool = True):
        return dict(sorted(val.items(), key=lambda kv: (kv[1], kv[0]), reverse=reserve))
    
    @staticmethod
    def sum(stats: dict):
        return sum(stats.values())
    
    @staticmethod
    def sum_items(val: dict):
        return dict(functools.reduce(operator.add, map(collections.Counter, list(val.values()))))

    @staticmethod
    def mean_items(val: dict):
        return {k: v / len(val) for k, v in dict(functools.reduce(operator.add, map(collections.Counter, list(val.values())))).items()}

    @staticmethod
    def add(stats: dict, val):
        return {k: v + val for k, v in stats.items()}

    @staticmethod
    def filter_target(val: dict, target: Union[str, List[str]]):
        fill = copy.deepcopy(val)
        if isinstance(target, list):
            for tar in target:
                fill.pop(tar)
        else:
            fill.pop(target)
        return fill
    
    @staticmethod
    def log_logistic_distribution(c: float, n: int, is_origin: bool = False):
        assert c >= 1.5 and n > 1
        x = np.linspace(1. / fisk.mean(c), fisk.ppf(0.99, c), n)
        ratio = fisk.pdf(x, c)
        return np.array(ratio) if is_origin else np.array(ratio) / np.sum(ratio)
    
    @staticmethod
    def get_items(stats: dict, keys: list):
        ret = {}
        for domain in stats.keys():
            ret[domain] = {}
            for cls in keys:
                ret[domain][cls] = stats[domain][cls]
        return ret
    
    def calc_percent(self, stats: dict, has_inner=False):
        if not has_inner:
            stats = {'inner': stats}
        
        percent = {}
        for domain in stats.keys():
            total, percent[domain] = self.sum(stats[domain]), {}
            for k, v in stats[domain].items():
                percent[domain][k] = v / total
            # end for k, v
        # end for domain

        return percent['inner'] if not has_inner else percent

    def calc_percent_items(self, stats: dict, keys: list = None, keep: bool = False):
        total = self.sum_items(stats)
        keys = list(stats.values()[0].keys()) if keys is None or len(keys) == 0 else keys

        percent = {}
        for domain in stats.keys():
            if len(keys) > 1 or keep:
                percent[domain] = {}
                for cls in keys:
                    percent[domain][cls] = stats[domain][cls] / total[cls]
                # end for cls
            else:
                percent[domain] = stats[domain][keys[0]] / total[keys[0]]
        # end for domain
        
        return percent
    
    def minus_valid(self, val: dict, check: bool = True):
        for domain in val.keys():
            val[domain] = self.add(val[domain], -self.valid)
            assert all(v > self.lower for v in val[domain].values()) if check else True
        return val
    
    def mise(self, domains: list, classes: list):
        mise = {}
        for target in domains:
            mise[target] = {}
            for source in domains:
                if source != target:
                    mise[target][source] = {name: 0 for name in classes}
            # end for source
        # end for target
        return mise
    
    @staticmethod
    def combine_dict(target: dict, source: dict, has_inner: bool = False):
        if not has_inner:
            target['inner'], source['inner'] = target, source

        for domain in source.keys():
            for cls in source[domain].keys():
                target[domain][cls] = int(source[domain][cls])

        return target if has_inner else target['inner']
    
    def generate(self, domains: list, stats: dict):
        raise NotImplemented
    
    def __str__(self):
        return self.__class__.__name__
    

class TotalHeavyTail(Generator):
    def __init__(self, valid, test, c: float = 3, lower: int = 1):
        super(TotalHeavyTail, self).__init__(valid, test, lower)
        assert 0 < self.test <= 1
        self.c = c

    def _set_value(self, expected, naive) -> int:
        ret = int(naive * (1 - self.test)) if expected >= naive else expected
        return ret if ret >= self.lower else self.lower

    def get_value_by_percent(self, num, percent, naive):
        ret, cls = {}, list(list(naive.values())[0].keys())[0]
        for domain in naive.keys():
            ret[domain] = {cls: self._set_value(int(num * percent[domain][cls]), naive[domain][cls])}
        return ret

    def generate(self, domains: list, stats: dict):
        stats, mise = self.minus_valid(stats), self.mise(domains, list(stats[domains[0]].keys()))
        ratio = self.log_logistic_distribution(self.c, len(stats[domains[0]]))
        for target in domains:
            source = self.filter_target(stats, target)
            each_cls_total = self.sorted_items(self.sum_items(source))
            cls_list = list(each_cls_total.keys())

            for i in range(len(cls_list)):
                if i == 0:
                    expected = (1 - self.test) * each_cls_total[cls_list[0]]
                    total = expected / ratio[0]
                else:
                    expected = total * ratio[i]
                    expected = each_cls_total[cls_list[i]] * (1 - self.test) if expected > each_cls_total[cls_list[i]] else expected

                percent = self.calc_percent_items(source, [cls_list[i]], keep=True)
                num_dict = self.get_value_by_percent(expected, percent, self.get_items(source, [cls_list[i]]))
                mise[target] = self.combine_dict(mise[target], num_dict, has_inner=True)
            # end for i
        # end for target
        return mise


class Cross(Generator):
    def __init__(self, valid, test, c: float = 3, middle: float = 1, lower: int = 1):
        super(Cross, self).__init__(valid, test, lower)
        assert 0 < self.test <= 1
        self.c = c
        self.middle_percent = middle

    def _get_minimize_num(self, stats) -> int:
        minimize = np.inf
        for val in stats.values():
            minimize = np.min([int((1 - self.test) * val), minimize])
        return int(minimize)

    def _set_num(self, total, ratio, num) -> int:
        return int(np.max([np.min([total * ratio, num * (1 - self.test)]), 1]))

    def generate(self, domains: list, stats: dict):
        stats, mise = self.minus_valid(stats), self.mise(domains, list(stats[domains[0]].keys()))
        for target in domains:
            source = self.filter_target(stats, target)
            each_domain_total = self.sorted_items({name: self.sum(domain) for name, domain in source.items()}, reserve=False)
            order, n_split = {'negative': [], 'middle': [], 'positive': []}, len(each_domain_total) // 2
            name, start = list(each_domain_total.keys()), len(each_domain_total) - 2 * n_split
            for ind in range(len(each_domain_total)):
                if ind < start:
                    order_key = 'middle'
                elif ind < start + n_split:
                    order_key = 'positive'
                else:
                    order_key = 'negative'
                order[order_key].append(name[ind])
            # end for ind

            # positive
            cls_sorted_list = None
            for ind, domain in enumerate(order['positive']):
                ratio = self.log_logistic_distribution(self.c * (ind + 1), len(stats[domain]))
                if cls_sorted_list is None:
                    cls_sorted_list = list(self.sorted_items(source[domain]).keys())
                total = source[domain][cls_sorted_list[0]] * (1 - self.test) / ratio[0]
                for i in range(len(cls_sorted_list)):
                    mise[target][domain][cls_sorted_list[i]] = self._set_num(total, ratio[i], source[domain][cls_sorted_list[i]])

            # negative
            for ind, domain in enumerate(order['negative']):
                ratio = self.log_logistic_distribution(self.c * (ind + 1), len(stats[domain]))
                total = source[domain][cls_sorted_list[-1]] * (1 - self.test) / ratio[0]
                for i in range(len(cls_sorted_list)):
                    mise[target][domain][cls_sorted_list[len(cls_sorted_list) - i - 1]] = \
                        self._set_num(total, ratio[i], source[domain][cls_sorted_list[len(cls_sorted_list) - i - 1]])

            # middle
            mean = np.mean(list(self.mean_items(self.filter_target(mise[target], order['middle'])).values())) * self.middle_percent
            for domain in order['middle']:
                num = int(np.min([self._get_minimize_num(source[domain]), mean]))
                for cls in mise[target][domain].keys():
                    mise[target][domain][cls] = num
        # end for target
        return mise


class Duality(Generator):
    def __init__(self, valid, test, c: float = 3, middle_range: list = [40, 50], lower: int = 1):
        super(Duality, self).__init__(valid, test, lower)
        assert 0 < self.test <= 1
        self.c = c
        self.middle_range = middle_range

    def _set_num(self, total, ratio, num) -> int:
        return int(np.max([np.min([total * ratio, num * (1 - self.test)]), 1]))

    def _beta_sample(self, a: float = 0.5, b: float = 0.5, n: int = 1):
        return np.random.beta(a, b, size=n)

    def generate(self, domains: list, stats: dict):
        stats, mise = self.minus_valid(stats), self.mise(domains, list(stats[domains[0]].keys()))
        for target in domains:
            source = self.filter_target(stats, target)
            each_domain_total = self.sorted_items({name: self.sum(domain) for name, domain in source.items()}, reserve=False)
            order, n_split = {'negative': [], 'middle': [], 'positive': []}, round(len(each_domain_total) / 3)
            name = list(each_domain_total.keys())
            for ind in range(len(each_domain_total)):
                if ind < n_split:
                    order_key = 'positive'
                elif ind < len(each_domain_total) - n_split:
                    order_key = 'negative'
                else:
                    order_key = 'middle'
                order[order_key].append(name[ind])

            cls_sorted_list, upper_bound = None, 0
            for ind, domain in enumerate(order['positive']):
                ratio = self.log_logistic_distribution(self.c * (ind + 1), len(stats[domain]))
                if cls_sorted_list is None:
                    cls_sorted_list = list(self.sorted_items(source[domain]).keys())
                total = source[domain][cls_sorted_list[0]] * (1 - self.test) / ratio[0]
                for i in range(len(cls_sorted_list)):
                    mise[target][domain][cls_sorted_list[i]] = self._set_num(total, ratio[i], source[domain][cls_sorted_list[i]])
                n_domain_sampling = self.sum(mise[target][domain])
                upper_bound = n_domain_sampling if upper_bound < n_domain_sampling else upper_bound
            # end for ind [positive]

            n_sampling = np.random.randint(self.middle_range[0], self.middle_range[1] + 1)
            n_sampling = len(cls_sorted_list) if n_sampling / len(cls_sorted_list) < 1 else n_sampling
            mean = int(math.ceil(n_sampling / len(cls_sorted_list)))
            for domain in order['middle']:
                for cls in mise[target][domain].keys():
                    mise[target][domain][cls] = int(np.min([mean, source[domain][cls]]))
            # end for domain [middle]

            for ind, domain in enumerate(order['negative']):
                n_domain_sampling = round(self._beta_sample()[0] * (upper_bound - n_sampling) + n_sampling)
                ratio = self.log_logistic_distribution(self.c * (ind + 1), len(stats[domain]))
                for i in range(len(cls_sorted_list)):
                    mise[target][domain][cls_sorted_list[len(cls_sorted_list) - i - 1]] = \
                        self._set_num(n_domain_sampling, ratio[i], source[domain][cls_sorted_list[len(cls_sorted_list) - i - 1]])
            # end for ind [negative]
        return mise


class ImbalancedDomains(Generator):
    def __init__(self, valid, test, lower: int = 1, mode: str = 'max', num: list = None):
        super().__init__(valid, test, lower)
        assert mode in ['max', 'manner']
        self.mode = mode
        self.num = num

    def _get_minimize_num(self, stats) -> int:
        minimize = np.inf
        for val in stats.values():
            minimize = np.min([int((1 - self.test) * val), minimize])
        return int(minimize)

    def generate(self, domains: list, stats: dict):
        stats, mise = self.minus_valid(stats), self.mise(domains, list(stats[domains[0]].keys()))
        for target in domains:
            source = self.filter_target(stats, target)
            for domain in source.keys():
                num = self._get_minimize_num(source[domain]) if self.mode == 'max' else self.num[target][domain]
                mise[target][domain] = {cls: num for cls in mise[target][domain].keys()}
        return mise

    def __str__(self):
        return super().__str__() + self.mode.capitalize()


class ImbalancedClasses(Generator):
    def __init__(self, valid, test, lower: int = 1, mode: str = 'max', upper: dict = None, c: float = 3, num: dict = None):
        super(ImbalancedClasses, self).__init__(valid, test, lower)
        assert mode in ['max', 'auto', 'manner']
        assert mode == 'max' or (mode == 'auto' and upper is not None) or (mode == 'manner' and num is not None)
        assert c > 1
        self.c = c
        self.mode = mode
        self.upper = upper
        self.num = num

    def _get_minimize_num(self, stats) -> int:
        minimize = np.inf
        for val in stats.values():
            minimize = np.min([int((1 - self.test) * val), minimize])
        return int(minimize)

    def _set_num(self, num, ratio):
        num = int(num * ratio)
        return num if num > 1 else 1

    def generate(self, domains: list, stats: dict):
        stats, mise = self.minus_valid(stats), self.mise(domains, list(stats[domains[0]].keys()))
        ratio = self.log_logistic_distribution(self.c, len(stats[domains[0]]))
        for target in domains:
            source = self.filter_target(stats, target)
            for domain in source.keys():
                if self.mode == 'max':
                    num = self._get_minimize_num(source[domain])
                elif self.mode == 'auto':
                    ind = np.random.permutation(ratio.shape[0])
                    num = {cls: self._set_num(self.upper[target], ratio[ind[i]]) for i, cls in enumerate(mise[target][domain].keys())}
                else:
                    raise NotImplemented('>_<')
                mise[target][domain] = {cls: (num if isinstance(num, int) else num[cls]) for cls in mise[target][domain].keys()}
            # end for domain
        # end for target
        return mise

    def __str__(self):
        return super().__str__() + self.mode.capitalize()


root, isshow = 'stats', True
# Total Heavy Tail
dataset, valid, test, many, few, c = 'pacs', 15, 0.2, 120, 6, 3
# dataset, valid, test, many, few, c = 'officehome', 5, 0.2, 30, 6, 3
# dataset, valid, test, many, few, c = 'vlcs', 5, 0.2, 200, 14, 3
# dataset, valid, test, many, few, c = 'domainnet', 5, 0.2, 300, 50, 2
generate = TotalHeavyTail(valid, test, c=c)

# Cross
# # ## PACS sketch thres -> {120, 30}
# dataset, valid, test, many, few, c, percent = 'pacs', 15, 0.2, 160, 60, 3, 0.3
# dataset, valid, test, many, few, c, percent = 'officehome', 5, 0.2, 20, 10, 3, 0.3
# dataset, valid, test, many, few, c, percent = 'vlcs', 5, 0.2, 200, 40, 3, 0.3
# generate = Cross(valid, test, c=c, middle=percent)
#
# # Duality
# # ## PACS photo and sketch thres -> {100, 15}
# dataset, valid, test, many, few, c, middle_range = 'pacs', 15, 0.2, 100, 30, 3, [40, 50]
# dataset, valid, test, many, few, c, middle_range = 'officehome', 5, 0.2, 19, 10, 3, [65, 70]
# dataset, valid, test, many, few, c, middle_range = 'vlcs', 5, 0.2, 100, 20, 3, [40, 50]
# dataset, valid, test, many, few, c, middle_range = 'domainnet', 5, 0.2, 60, 20, 2, [40, 50]
# generate = Duality(valid, test, c=c, middle_range=middle_range)

# PACS sample
## domain imbalanced, but class balanced
# dataset, valid, test, many, few = 'pacs', 15, 0.2, 0, 0
# num = {
#     'art_painting': {'cartoon': 20, 'photo': 100, 'sketch': 4},
#     'cartoon': {'art_painting': 4, 'photo': 100, 'sketch': 20},
#     'photo': {'art_painting': 100, 'cartoon': 4, 'sketch': 20},
#     'sketch': {'art_painting': 100, 'cartoon': 4, 'photo': 20},
# }
# num = {
#     'art_painting': {'cartoon': 20, 'photo': 4, 'sketch': 100},
#     'cartoon': {'art_painting': 100, 'photo': 4, 'sketch': 20},
#     'photo': {'art_painting': 4, 'cartoon': 100, 'sketch': 20},
#     'sketch': {'art_painting': 4, 'cartoon': 100, 'photo': 20},
# }
# generate = ImbalancedDomains(valid, test, mode='max')
# generate = ImbalancedDomains(valid, test, mode='manner', num=num)

# ## class imbalanced, but domain balanced
# dataset, valid, test, many, few, c = 'pacs', 15, 0.2, 10, 3, 4
# generate = ImbalancedClasses(valid, test, mode='max')
# upper = {'art_painting': 52, 'cartoon': 52, 'photo': 52, 'sketch': 52}
# generate = ImbalancedClasses(valid, test, mode='auto', upper=upper, c=c)
#


root = os.path.join(root, DATASETS[dataset])
if not os.path.exists(root):
    os.makedirs(root)
# filestream = open(os.path.join(root, str(generate) + '.py'), 'w')
filestream = sys.stdout

stats = main(dataset, valid, many, few, generate, file=filestream)
draw_stats(stats, None if isshow else '%s.pdf' % (str(generate.lower())))
