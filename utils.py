import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def get_targets(boxes):
    ''''''
    'Function for casting boxes info into suitable for Faster-RCNN format'
    
    targets = []
    for i in range(len(boxes)): 
        d = {}
        d['boxes'] = boxes[i][1:][None, ...]
        d['labels'] = boxes[i][0][None, ...].to(torch.int64)
        targets.append(d)
    return targets

def visualize(dataset, i=5):
    ''''''''
    'Visualizes selected element from dataset'
    fig, ax = plt.subplots(1, figsize=(12,9))
    _, __ = dataset[i]
    ax.imshow(_.transpose(1,2,0))
    c, x1, y1, x2, y2 = __
    w = x2 - x1
    h = y2 - y1
    bbox = patches.Rectangle((x1,y1), w , h ,
                linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(bbox)
    plt.axis('off')
    print(__)
    plt.show()