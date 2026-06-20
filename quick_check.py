import os

def check():
    path = 'data/input/TEST.wav'
    if os.path.exists(path):
        print('STATUS: File TEST.wav found!')
    else:
        print('STATUS: File NOT found!')

    out_dir = 'data/output'
    if os.path.exists(out_dir):
        print(f'STATUS: Output directory exists. Content: {os.listdir(out_dir)}')
    else:
        print('STATUS: Output directory does not exist.')

if __name__ == '__main__':
    check()
