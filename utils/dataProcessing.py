from makeAugmentImage import aug_img_save
from makeOriginImage import check_img_save_origin
from conf import *

train_df = pd.read_csv(TRAIN_CSV)

check_img_save_origin(train_df, show_num=5, save_origin=True)