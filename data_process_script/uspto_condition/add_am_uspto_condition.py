import argparse
import shutil
import pandas
import random
import os
import json
import multiprocessing
import subprocess
import sys


def get_temp_dir():
    res = [chr(random.randint(ord('a'), ord('z'))) for x in range(15)]
    return ''.join(res)


def map_shapre(fname, odir, bs, Rs, Lk):
    Lk.acquire()
    for k, v in Rs.items():
        if v:
            Rs[k] = False
            device = k
            break
    Lk.release()

    command = [
        sys.executable,
        "map_augx.py",
        "--input", fname,
        "--dir", odir,
        "--bs", str(bs)
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)

    print('executing: ', ' '.join(command))
    subprocess.run(command, env=env)

    Lk.acquire()
    Rs[device] = True
    Lk.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, required=True)
    parser.add_argument('--share_size', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument(
        '--vis_devices', type=str, default='',
        help='CUDA_VISIBLE_DEVICE'
    )
    args = parser.parse_args()

    raw_info = pandas.read_csv(args.input_file)

    if args.vis_devices == '':
        VDS = list(range(args.num_gpus))
    else:
        VDS = [int(x) for x in args.vis_devices.split(',')]
        assert len(VDS) == args.num_gpus, \
            "Devices Ids mismatch num of devices"

    temp_dir = get_temp_dir()

    while os.path.exists(temp_dir):
        temp_dir = get_temp_dir()

    os.makedirs(temp_dir)

    ilist, ppargs, all_len = {}, [], len(raw_info)

    Po = multiprocessing.Pool(processes=args.num_gpus)
    Mn = multiprocessing.Manager()
    Rs, Lk = Mn.dict(), Mn.Lock()

    for x in VDS:
        Rs[x] = True

    for idx in range(all_len):
        ilist[idx] = raw_info.loc[idx]['canonical_rxn']
        if len(ilist) == args.share_size or idx == all_len - 1:
            Fname = f'{idx - len(ilist) + 1}-{idx}.json'
            with open(os.path.join(temp_dir, Fname), 'w') as Fout:
                json.dump(ilist, Fout)
            ilist = {}
            ppargs.append((Fname, temp_dir, args.batch_size, Rs, Lk))

    Po.starmap(map_shapre, ppargs)
    Po.close()
    Po.join()

    all_info = {}
    for x in ppargs:
        with open(os.path.join(temp_dir, f'out_{x[0]}'), 'r') as Fin:
            INFO = json.load(Fin)
        all_info.update(INFO)

    mapped_res, map_conf = [], []
    for idx in range(all_len):
        mapped_res.append(all_info[str(idx)]['mapped_rxn'])
        map_conf.append(all_info[str(idx)]['confidence'])

    raw_info['mapped_rxn'] = mapped_res
    raw_info['confidence'] = map_conf

    raw_info.to_csv(args.output_file)
    shutil.rmtree(temp_dir)
