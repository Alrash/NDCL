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
    'officehome': 'OfficeHome'
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
}

NCOLS = {
    DATASETS['pacs']: 4,
    DATASETS['vlcs']: 5,
    DATASETS['officehome']: 10,
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
        indices = (ind) if folds == 1 else (ind // folds, ind % folds)
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