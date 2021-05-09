import os
import shutil
import cv2
import numpy as np
import optparse

from tqdm import tqdm

parser = optparse.OptionParser()

parser.add_option(
    '-i',
    '--input',
    default=None,
    dest='input_folder',
    help='Input folder')
parser.add_option(
    '-o', '--output',
    default=None,
    dest='output_folder',
    help='Output folder')
options, args = parser.parse_args()

# define patches
INPUT_PATH = options.input_folder
OUTPUT_PATH = options.output_folder

# make neecessary folder
os.makedirs(OUTPUT_PATH, exist_ok=True)

# get img list
img_list = os.listdir(INPUT_PATH)
# print(img_list)

for img_name in tqdm(img_list):
    img_path = os.path.join(INPUT_PATH,img_name)
    small_token_list = os.listdir(img_path)
    # print(small_token_list)
    for small_token_name in small_token_list:
        token_path = os.path.join(img_path, small_token_name)
        small_token = cv2.imread(token_path)
        small_width = small_token.shape[1]
        small_height = small_token.shape[0]

        big_token = np.zeros((100,400,3))
        big_width = big_token.shape[1]
        big_height = big_token.shape[0]

        alpha = big_width/small_width
        beta = big_height/small_height
        # print("\nalpha = ", alpha)
        # print("beta = ", beta)
        gamma = min(alpha, beta)
        # print("gamma = ", gamma)
        # print("small_token_shape_after = ",small_token.shape)
        # cv2.imshow("small_token_shape_after",small_token)
        # cv2.waitKey(0)
        small_token = cv2.resize(small_token,None,fx=gamma,fy=gamma)
        small_width = small_token.shape[1]
        small_height = small_token.shape[0]
        # cv2.imshow("small_token_shape_before",small_token)
        # cv2.waitKey(0)

        # print("big_token_shape = ",big_token.shape)
        # print("small_token_shape_before = ",small_token.shape)

        # if (small_width % 2 != 0):
        #     print("width le")
        #     small_token.resize(small_height,small_width-1,3)
        #     small_width = small_token.shape[1]
        # if (small_height % 2 != 0):
        #     print("height le")
        #     small_token.resize(small_height-1, small_width,3)
        #     small_height = small_token.shape[0]
        # print("small_token_shape_nomalized = ",small_token.shape)
        # cv2.imshow("small_token_shape_nomalized",small_token)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        x1 = int(200+small_width/2)
        y1 = int(50+small_height/2)
        x2 = int(200-small_width/2)
        y2 = int(50-small_height/2)
        # print(y2,":",y1)
        # print(x2,":",x1)
        # print(big_token[x2:x1, y2:y1].shape)
        # print(big_token[y2:y1, x2:x1].shape)
        # big_token[x2:x1,y2:y1] = small_token
        big_token[y2:y1, x2:x1] = small_token

        out_img = os.path.join(OUTPUT_PATH,img_name)
        os.makedirs(out_img, exist_ok=True)
        out_fn = os.path.join(OUTPUT_PATH,img_name,small_token_name)
        cv2.imwrite(out_fn, big_token)