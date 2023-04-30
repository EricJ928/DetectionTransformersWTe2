import os
import glob
import shutil

# dirt = r'G:\OneDrive - Nanyang Technological University\Eric\Projects\AI_STM\Au_example\2021-2022-select\40nm_64Pix'
# dirt = r'G:\OneDrive - Nanyang Technological University\Eric\Projects\AI_STM\Au_example\2021-2022-select\30nm_64Pix'
dirt = r'C:\Users\Eric JIA\OneDrive - Nanyang Technological University\Eric\Projects\AI_STM\Au_example\2020-select\40nm_64Pix'
os.chdir(dirt)
target_folder = r'slice'
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

fnames = glob.glob('*[0-9].npy')
print('No. files:', len(fnames))
# print(fnames)

for fn in fnames:
    # shutil.copy2(fn, target_folder)
    shutil.move(fn, target_folder)
