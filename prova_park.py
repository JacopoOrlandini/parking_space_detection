"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 as cv
import numpy as np
from merge import HoughBundler
from time import time
def main(argv):
    

    start = time()
    default_file = 'park.jpg'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    kernel = np.ones((5,5),np.float32)/25
    blur = cv.filter2D(src,-1,kernel)
    dst = cv.Canny(blur, 160,230, None, 3)
    MO_kernel = np.ones((9,9),np.uint8)
    dst = cv.morphologyEx(dst, cv.MORPH_CLOSE, MO_kernel)
    cv.imshow("closing image", dst)

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    # Decomment to draw filtering image below Hough lines
    cdstP = np.copy(cdst)
    #Use just a black mat
    #cdstP = np.zeros((cdst.shape[0],cdst.shape[1],3))
    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(cdst, pt1, pt2, (0,i,255), 3, cv.LINE_AA)
    
    

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 100, None, 80, 10)
    print("Size linee HLP :" + str(len(linesP)))
    a = HoughBundler()
    foo = a.process_lines(linesP, dst)
    print("Size linee bundled HLP :" + str(len(foo)))
    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0] 
    #         cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,i*5+10,255), 1, cv.LINE_AA)
    marked = []
    max_degree = []
    min_distance = 20

    if foo is not None:
        for i in range(0, len(foo)):
            l = foo[i]
            '''
            Documentazione provvisoria
            (xStart,yStart,xEnd,yEnd)    
            '''
            line = []
            line.append(l[0][0])
            line.append(l[0][1])
            line.append(l[1][0])
            line.append(l[1][1])
            # Descriptor for line (values) and orientation
            print(line)
            is_alone = True
            if (marked == []):
                cv.line(cdstP, (l[0][0],l[0][1]), (l[1][0],l[1][1]), (10,10,255), 2, cv.LINE_AA)
                cv.circle(cdstP,(line[0],line[1]), 5, (255,0,0), -1)
                cv.circle(cdstP,(line[2],line[3]), 5, (0,0,255), -1)
                max_degree = a.get_orientation(line)
                marked.append(line)
            else:
                for j in marked:
                    if(a.get_distance(j,line)<min_distance and a.get_orientation(line)< max_degree+20):
                        is_alone=False
                        break
            #TODO qua condizione line[0] > 5 capire il problema di fondo?
            # Collega dei punti molto distanti per x ma vicini per y -> SBAGLAITO
            if is_alone and line[0] > 5:
                cv.line(cdstP, (l[0][0],l[0][1]), (l[1][0],l[1][1]), (10,10,255), 2, cv.LINE_AA)
                marked.append(line)
                #Draw starting and ending point of Hough line
                cv.circle(cdstP,(line[0],line[1]), 5, (255,0,0), -1)
                cv.circle(cdstP,(line[2],line[3]), 5, (0,0,255), -1)

    cv.imshow("Source", src)
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    cdstP = cdstP.astype(np.uint8)
    img = cv.cvtColor(cdstP, cv.COLOR_RGB2GRAY)
    #contours, hierarchy = cv.findContours(cdstP, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #cv.imshow("Window", img)

    print("durata = " +str(time()-start))
    cv.waitKey()
    return 0
    
if __name__ == "__main__":
    main(sys.argv[1:])
