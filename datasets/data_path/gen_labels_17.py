import os.path as osp
import os
import numpy as np


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


# Update this to your MOT17 root (the folder containing MOT17/)
mot_root = '/kaggle/working/MOTR'

seq_root = osp.join(mot_root, 'MOT17/images/train')

# Write labels to a writable location, then symlink into MOT17/
label_root = '/kaggle/working/MOT17_labels/train'
mkdirs(label_root)

# Create symlink so training code finds labels_with_ids inside MOT17/
symlink_target = osp.join(mot_root, 'MOT17/labels_with_ids')
if not osp.exists(symlink_target):
    os.symlink('/kaggle/working/MOT17_labels', symlink_target)
    print(f"Created symlink: {symlink_target} -> /kaggle/working/MOT17_labels")

seqs = [s for s in os.listdir(seq_root) if 'SDP' in s]

tid_curr = 0
tid_last = -1
for seq in sorted(seqs):
    seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
    idx = np.lexsort(gt.T[:2, :])
    gt = gt[idx, :]

    seq_label_root = osp.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)

    for fid, tid, x, y, w, h, mark, label, vis in gt:
        if mark == 0 or not label == 1:
            continue
        fid = int(fid)
        tid = int(tid)
        if not tid == tid_last:
            tid_curr += 1
            tid_last = tid
        x += w / 2
        y += h / 2
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
        with open(label_fpath, 'a') as f:
            f.write(label_str)

print("Done. Generated labels_with_ids for MOT17.")
