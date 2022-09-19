from os import rmdir
from pathlib import Path
import tqdm
import shutil


target_dir = ['doc/', 'lib/', 'pointnet2_lib/', 'pretrain_pth/', 'rank_0/', 'tests/', 'tools/']
rm_dir = ["rank_0","__pycache__","dist","build"]

# print(Path(__file__).absolute().parent.parent.absolute())
# li = Path(__file__).absolute().parent.parent.absolute().glob("**/*")
for dirpath  in target_dir:
    li = Path(dirpath).absolute().glob("**/*")
    for i in li:
        if i.name=="kitti" or i.name == "data":
            continue
        if i.is_dir():
            try:
                if i.name in rm_dir:
                    print(i.absolute())
                    shutil.rmtree(i)
                if i.name.split(".")[-1] == "egg-info":
                    print(i.absolute())
                    shutil.rmtree(i)
            except:
                pass
            

shutil.rmtree('.pytest_cache/')

# li = Path(__file__).absolute().parent.parent.absolute().glob("*")
# ans = []
# for i in li:
#     if i.is_dir():
#         if i.name in ["data",".git","kitti"]:
#             continue
#         # print(i)
#         ans.append(str(i))
# print(ans)