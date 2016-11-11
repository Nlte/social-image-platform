import os
import pandas as pd
import numpy as np
import re
from fnmatch import fnmatch

import json

IMAGE_DIR = 'data/images'
ANNOTATION_DIR = 'data/annotations'
TAG_DIR = 'data/meta/tags'

def build_annotation_json():
    ids_to_images = {}
    for f in os.listdir(IMAGE_DIR):
        if fnmatch(f, '.*'):
            continue
        idx = int(re.findall('\d+', f)[0])
        ids_to_images[idx] = f

    ids_to_tags = {}
    for f in os.listdir(TAG_DIR):
        if fnmatch(f, '.*'):
            continue
        idx = int(re.findall('\d+', f)[0])
        f_path = os.path.join(TAG_DIR, f)
        with open(f_path, 'r') as tagfile:
            content = tagfile.read().split('\r\n')
        content = [c for c in content if c != '']
        ids_to_tags[idx] = content

    annotations_to_ids = {}
    for f in os.listdir(ANNOTATION_DIR):
        if fnmatch(f, '.*'):
            continue
        label = f.split('.')[0]
        annot_path = os.path.join(ANNOTATION_DIR, f)
        with open(annot_path, 'r') as annotfile:
            content = annotfile.read().split('\r\n')
        content = [c for c in content if c != '']
        content_int = [int(c) for c in content]
        annotations_to_ids[label] = content_int

    ids_to_annotations = {}
    for i in ids_to_images.keys():
        ids_to_annotations[i] = [k for k, v in annotations_to_ids.items() if i in v]

    for key, value in ids_to_annotations.items():
        new_value = [v.split('_')[0] for v in value]
        new_value = list(set(new_value))
        ids_to_annotations[key] = new_value

    json_objects = []
    for i in ids_to_images.keys():
        obj = {'image':ids_to_images[i], 'annotation':ids_to_annotations[i], 'tags':ids_to_tags[i]}
        json_objects.append(obj)

    with open('annotations.json', 'w') as f:
        json.dump(json_objects, f)

def build_exif_json():
    


if __name__ == '__main__':
    build_annotation_json()
