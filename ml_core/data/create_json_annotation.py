import os
import pandas as pd
import numpy as np
import re
import sys
from datetime import datetime
from fnmatch import fnmatch
import pdb

import json

IMAGE_DIR = "mirflickr"
ANNOTATION_DIR = "annotation"

def main():
    print("%s - Creating image ids" % datetime.now())
    ids_to_images = {}
    n = len(os.listdir(IMAGE_DIR))
    for i, f in enumerate(os.listdir(IMAGE_DIR), 1):
        if fnmatch(f, '.*'):
            continue
        if os.path.isdir(os.path.join(IMAGE_DIR, f)):
            continue
        im_id = ''.join(c for c in f.split('.')[0] if c.isdigit())
        ids_to_images[im_id] = f
        sys.stdout.write(' >> %d / %d\r' % (i, n))
        sys.stdout.flush()

    print("%s - Creating annotations to ids" % datetime.now())
    annotations_to_ids = {}
    n = len(os.listdir(ANNOTATION_DIR))
    for i, f in enumerate(os.listdir(ANNOTATION_DIR)):
        if fnmatch(f, '.*') or fnmatch(f, 'README*'):
            continue
        label = f.split('.')[0]
        annot_path = os.path.join(ANNOTATION_DIR, f)
        with open(annot_path, 'r', encoding='utf8') as annotfile:
            content = annotfile.readlines()
        content = [c.strip() for c in content if c != '']
        annotations_to_ids[label] = content
        sys.stdout.write(' >> %d / %d\r' % (i, n))
        sys.stdout.flush()

    print("%s - Creating ids to annotations" % datetime.now())
    ids_to_annotations = {}
    n = len(ids_to_images)
    for i, j in enumerate(ids_to_images.keys()):
        labels = [k for k, v in annotations_to_ids.items() if j in v]
        labels = [l.split('_')[0] for l in labels] # remove '_r1', '_r2'
        ids_to_annotations[j] = list(set(labels)) # remove possible duplicates
        sys.stdout.write(' >> %d / %d\r' % (i, n))
        sys.stdout.flush()

    print("%s - Dumping into annotations.json" % datetime.now())
    json_objects = []
    n = len(ids_to_images)
    for i, j in enumerate(ids_to_images.keys()):
        obj = {'image':ids_to_images[j], 'annotation':ids_to_annotations[j]}
        json_objects.append(obj)
        sys.stdout.write(' >> %d / %d\r' % (i, n))
        sys.stdout.flush()

    with open('annotations.json', 'w') as f:
        json.dump(json_objects, f)


if __name__ == '__main__':
    main()
