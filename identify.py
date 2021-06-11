# coding:utf-8
from os import path, makedirs, listdir
from argparse import ArgumentParser, RawTextHelpFormatter
from datetime import datetime
from copy import deepcopy
import numpy as np
import cv2
from cv2 import imread, imwrite, blur, cvtColor, COLOR_BGR2GRAY, HoughCircles, HOUGH_GRADIENT, rectangle, circle, putText, FONT_HERSHEY_COMPLEX_SMALL
import time


def GammaCorrect_CIETrans(rgblist):
    gamma, alpha1, alpha2 = 2.4, 0.055, 1.055
    thres = 0.04045
    m_matrix = np.array([[0.4124564,0.3575761,0.1804375],[0.2126729,0.7151522,0.0721750],[0.0193339,0.1191920,0.9503041]],dtype=np.float)

    gamma_arr = np.zeros((3,1),dtype=np.float)
    '''
    Gamma correction:
    '''
    for i in range(len(rgblist)):
        if float(rgblist[i]) > thres:
            gamma_arr[i] = np.power((float(rgblist[i])/255. + alpha1) / alpha2, gamma)
        else:
            gamma_arr[i] = (float(rgblist[i])/255.) / 12.92
    '''
    Corrected RGB to XYZ:
    '''
    XYZ_arr = 100 * np.dot(m_matrix, gamma_arr)
    '''
    XYZ to xyY:
    '''
    xyz_s_arr = (1 / (XYZ_arr[0]+XYZ_arr[1]+XYZ_arr[2])) * XYZ_arr
    # out_x, out_y, out_Luminance:
    return xyz_s_arr[0], xyz_s_arr[1], XYZ_arr[1]

def gamma_streching(img,gamma):
    img = np.array(img/255.0, dtype=np.float)
    img = np.power(img, gamma)
    out = np.array(img*255.0, "uint8")
    return out

def proceed(params):

    in_dir = str(params.in_dir)
    if not str(params.out_dir):
        out_dir = str(params.out_dir)
    else:
        out_dir = str(datetime.now().strftime("%y-%m-%d_%H-%M-%S"))
    if not path.exists(out_dir):
        makedirs(out_dir)
    
    for imgfile in listdir(in_dir):
        in_path = in_dir + '\\' + str(imgfile)
        print('Processing:', in_path)
        out_picC_path = out_dir + '\\' + str(imgfile).split('.')[0] + '_Circ.tif'
        out_picR_path = out_dir + '\\' + str(imgfile).split('.')[0] + '_Rect.tif'
        # out_txtC_path = out_dir + '\\' + str(imgfile).split('.')[0] + '_Circ.txt'
        out_txtR_path = out_dir + '\\' + str(imgfile).split('.')[0] + '_Rect.txt'
        pic = imread(in_path)
        picR = deepcopy(pic)
        picC = deepcopy(pic)
        pic_arr = np.array(pic, dtype=np.uint8)

        '''
        Extract Green_band / Mix_band
        '''
        # picG = deepcopy(pic)
        # for i in range(pic.shape[2]):
        #     picG[:,:,i] = pic_arr[:,:,1]
        # Mix Color band with Green and Blue in Red band.
        picMix = deepcopy(pic)
        picMix[:,:,2] = (pic_arr[:,:,0] * 0.7 + pic_arr[:,:,1] * 0.2)

        ''' 
        Process gray_pic for circle detection
        '''
        # blur_pic = cv2.blur(picG, (10,10))
        blur_pic = blur(picMix, (10,10))
        # blur_pic = cv2.bilateralFilter(picMix, 10, sigmaSpace = 75, sigmaColor =75)
        blur_pic = gamma_streching(blur_pic,0.6)
        gray_pic = cvtColor(blur_pic, COLOR_BGR2GRAY)

        print('    Conducting HoughCircles.')
        circles= HoughCircles(
            gray_pic,
            HOUGH_GRADIENT,
            1,
            int(params.min_circle_distance), #50, #min circle distance (pixels)
            param1=int(params.edge_detect_thres), #Edge HIGH range param 
            param2=int(params.roundness_thres), #Roundness param
            minRadius=int(params.min_circleRadius), #min circle radius
            maxRadius=int(params.max_circleRadius) #max circle radius
        )
        print('    Num of detected circles: ', len(circles[0]))

        time_start = time.time()
        i = 1
        # out_fileC = open(out_txtC_path, 'w')
        # out_fileC.write('ID   x     y     r     R      G      B    CIE_x CIE_y CIE_Luminance\n')
        out_fileR = open(out_txtR_path, 'w')
        out_fileR.write('ID   x     y     r     R      G      B    CIE_x CIE_y CIE_Luminance\n')
        for circle in circles[0]:
            x, y, r = int(circle[0]), int(circle[1]), int(circle[2])

            '''
            Extract Circle's mean RGB value.
            '''
            # Fy, Fx = np.ogrid[:pic_arr.shape[0], :pic_arr.shape[1]]
            # mask = np.sqrt((Fx-x)*(Fx-x) + (Fy-y)*(Fy-y)) <= r
            # '''Method 1'''
            # mask = np.where(mask==True,1,0).astype(np.uint8)
            # sumb, sumg, sumr = np.sum(np.multiply(pic_arr[:,:,0],mask)), np.sum(np.multiply(pic_arr[:,:,1],mask)), np.sum(np.multiply(pic_arr[:,:,2],mask))
            # ave_b, ave_g, ave_r = sumb/np.sum(mask==1), sumg/np.sum(mask==1), sumr/np.sum(mask==1)
            # '''Method 2'''
            # mask = np.where(mask==True)
            # sumarr = np.zeros((3),dtype=np.uint64)
            # for j in range(mask[0].shape[0]):
            #     sumarr = sumarr + pic_arr[mask[0][j],mask[1][j],:]
            # Circ_ave_b, Circ_ave_g, Circ_ave_r = sumarr / int(mask[0].shape[0])
            # '''Method 3'''
            # mask = np.where(mask==True,False,True)
            # mask_arr_b, mask_arr_g, mask_arr_r = np.ma.masked_array(pic_arr[:,:,0], mask=mask, fill_value=999999), np.ma.masked_array(pic_arr[:,:,1], mask=mask, fill_value=999999), np.ma.masked_array(pic_arr[:,:,2], mask=mask, fill_value=999999)
            # ave_b, ave_g, ave_r = mask_arr_b.mean(), mask_arr_g.mean(), mask_arr_r.mean()

            '''
            Extract Rectangle's mean RGB value.
            '''
            rect = pic_arr[y-int(r/2):y+int(r/2),x-int(r/2):x+int(r/2),:]
            Rect_ave_b, Rect_ave_g, Rect_ave_r = np.average(rect[:,:,0]),np.average(rect[:,:,1]),np.average(rect[:,:,2])
            
            # print('%i: Circle: R:%.2f, G:%.2f, B:%.2f; Rectangle: R:%.2f, G:%.2f, B:%.2f'%(i, Circ_ave_r, Circ_ave_g, Circ_ave_b, Rect_ave_r, Rect_ave_g, Rect_ave_b))
            print('    %i: Rectangle: R:%.2f, G:%.2f, B:%.2f'%(i, Rect_ave_r, Rect_ave_g, Rect_ave_b))

            '''
            Gamma correction & CIE transfer.
            '''
            # Circ_CIE_x, Circ_CIE_y, Circ_CIE_Luminance = GammaCorrect_CIETrans([Circ_ave_r, Circ_ave_g, Circ_ave_b])
            Rect_CIE_x, Rect_CIE_y, Rect_CIE_Luminance = GammaCorrect_CIETrans([Rect_ave_r, Rect_ave_g, Rect_ave_b])
            
            '''
            Write result in txt.
            '''
            # out_fileC.write('%2d %5d %5d %4d %6.2f %6.2f %6.2f  %.2f  %.2f  %.2f\n'%(i, x, y, r, Circ_ave_r, Circ_ave_g, Circ_ave_b, Circ_CIE_x, Circ_CIE_y, Circ_CIE_Luminance))
            out_fileR.write('%2d %5d %5d %4d %6.2f %6.2f %6.2f  %.2f  %.2f  %.2f\n'%(i, x, y, r, Rect_ave_r, Rect_ave_g, Rect_ave_b, Rect_CIE_x, Rect_CIE_y, Rect_CIE_Luminance))
            
            '''
            Draw Circle.
            '''
            cv2.circle(picC, (x,y), r, (0,0,255), 3)
            # cv2.putText(pic, '%2d'%(i), (x+int(r/2),y+r), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (0, 0, 255))
            '''
            Draw Rectangle
            '''
            sx1, sx2 = x-int(r/2), x+int(r/2)
            sy1, sy2 = y-int(r/2), y+int(r/2)
            cv2.rectangle(picR, (sx1, sy1), (sx2, sy2), (0, 0, 255), 3)
            if (sx1 > 10):
                # cv2.putText(pic, '%2d: (%.2f, %.2f, %.2f)'%(i, CIE_x, CIE_y, CIE_Luminance), (int(sx1),int(sy1-6)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (255, 255, 255))
                putText(picR, '%2d'%(i), (int(sx1),int(sy1-6)), FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 255, 255))
            else:
                # cv2.putText(pic, '%2d: (%.2f, %.2f, %.2f)'%(i, CIE_x, CIE_y, CIE_Luminance), (int(sx1),int(sy1+15)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (255, 255, 255))
                putText(picR, '%2d'%(i), (int(sx1),int(sy1+15)), FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 255, 255))
            
            i += 1
        time_end = time.time()
        print('Time cost:', (time_end-time_start))
        imwrite(out_picC_path, picC)
        imwrite(out_picR_path, picR)
        

def run():
    '''
    The main function
    '''
    # Parse parameters
    parser = ArgumentParser(
        description='Detect Bacteria circles\' location from biochip and output its CIE values in a rectangle area.',
        epilog='Developed by xltan, contact me at xl_tan@whu.edu.cn',
        formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--in_dir',
        help='input directory of images.',
        required=True)
    parser.add_argument(
        '-o', '--out_dir',
        help='output dir of detected images.')
    parser.add_argument(
        '-d', '--min_circle_distance',
        help='Minimum distance of adjacent circles(pixels).(相邻圆之间的最小圆心距离(像素))',
        type=int,
        default=60)
    parser.add_argument(
        '-e', '--edge_detect_thres',
        help='Contrast threshold between circle edge and background.(圆边界与背景间的对比度阈值,值越高,对比度要求越高)',
        type=int,
        default=26)
    parser.add_argument(
        '-r', '--roundness_thres',
        help='Roundness threshold of circles.(圆度阈值,值越高圆度要求越高)',
        type=int,
        default=31)
    parser.add_argument(
        '--min_circleRadius',
        help='Minimum of circle radius.(检测圆的最小半径)',
        type=int,
        default=20)
    parser.add_argument(
        '--max_circleRadius',
        help='Maximum of circle radius.(检测圆的最大半径)',
        type=int,
        default=90)

    params = parser.parse_args()
    proceed(params)


if __name__ == '__main__':
    run()



