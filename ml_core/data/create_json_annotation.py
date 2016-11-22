import os
import pandas as pd
import numpy as np
import re
from datetime import datetime
from fnmatch import fnmatch

import json

IMAGE_DIR = "mirflickr"
ANNOTATION_DIR = "annotation"

def main():
    print("%s - Creating image ids" % datetime.now())
    ids_to_images = {}
    for f in os.listdir(IMAGE_DIR):
        if fnmatch(f, '.*'):
            continue
        if os.path.isdir(os.path.join(IMAGE_DIR, f)):
            continue
        idx = int(re.findall('\d+', f)[0])
        ids_to_images[idx] = f

    print("%s - Creating annotations to ids" % datetime.now())
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

    print("%s - Creating ids to annotations" % datetime.now())
    ids_to_annotations = {}
    for i in ids_to_images.keys():
        ids_to_annotations[i] = [k for k, v in annotations_to_ids.items() if i in v]

    for key, value in ids_to_annotations.items():
        new_value = [v.split('_')[0] for v in value]
        new_value = list(set(new_value))
        ids_to_annotations[key] = new_value

    print("%s - Dumping into annotations.json" % datetime.now())
    json_objects = []
    for i in ids_to_images.keys():
        obj = {'image':ids_to_images[i], 'annotation':ids_to_annotations[i]}
        json_objects.append(obj)

    with open('annotations.json', 'w') as f:
        json.dump(json_objects, f)


if __name__ == '__main__':
    main()
