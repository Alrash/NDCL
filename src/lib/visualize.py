import os

import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

import torch

__all__ = ['visualize_embedding']

_DATASET = {
    'pacs': ['Art Painting', 'Cartoon', 'Photo', 'Sketch'],
    'office-home': ['Art', 'Clipart', 'Product', 'Real World'],
}

# style of visualization image
_SHAPE = ['o', 'x', 's', 'D', '*', 'X']
_COLOR = 'Spectral'

# choice of visualization image
_IS_SHORT = False
_DISPLAY_AXIS = True
_LIMITED = False


def visualize_embedding(model, loader, dataset, test_env: list, save_dir: str, device,
                        task: str = 'domain_generalization'):
    if task.lower() != 'domain_generalization':
        raise NotImplementedError('Task %s has not implemented! >_<' % task)

    if dataset.lower() == 'pacs':
        envs = _DATASET[dataset.lower()]
    elif 'office' in dataset.lower() and 'home' in dataset.lower() and '31' not in dataset:
        dataset = 'office-home'
        envs = _DATASET[dataset.lower()]
    else:
        raise KeyError('Not find dataset %s when visualizing ! >_<' % dataset)

    n_envs, x, y, d, names = len(envs), [], [], [], []
    with torch.no_grad():
        for domain in range(n_envs):
            if domain in test_env:
                names.append((''.join([word[0] for word in envs[domain].split(' ')])).upper() if _IS_SHORT else envs[domain])
                if _LIMITED:
                    continue
            # end if
            x.append([]), y.append([])
            for (data, target) in loader[domain]:
                fea = model.embedding(data.to(device)).detach().cpu().clone().numpy()
                target = target.detach().cpu().clone().numpy()
                x[-1].append(fea), y[-1].append(target)
            # end for (data, target)
            x[-1], y[-1] = np.concatenate(x[-1], axis=0), np.concatenate(y[-1], axis=0)
            d.append(domain * np.ones(x[-1].shape[0]))
        # end for domain
    # end with
    x, y, d = np.concatenate(x, axis=0), np.concatenate(y, axis=0), np.concatenate(d, axis=0)

    filename = os.path.join(save_dir, '%s_%s.png' % (dataset.lower(), '#'.join(names)))


def _visualize(x, y, d, test_envs, filename, display_axis):
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    x = tsne.fit_transform(x)
    pass
