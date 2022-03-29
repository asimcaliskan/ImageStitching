from sift import SIFT
from orb import ORB
from surf import SURF
from cv2 import imread, cvtColor, COLOR_BGR2GRAY


if __name__ == "__main__":
    image_array = []
    gray_image_array = []
    #TODO: This code must be changed based on max and min image id.
    #TODO: Add a console UI to control the sw
    for ix in range(3, 7):
        if ix == 2:
            pass
        image_file_name = "dataset/cyl_image" + ( "0" + str(ix)  if ix < 10 else str(ix)) + ".png"
        image_array.append(imread(image_file_name))  
        gray_image_array.append(cvtColor(image_array[-1], COLOR_BGR2GRAY))      
    
    #DECLARATIONS BEGIN
    sift = SIFT(image_array=image_array, gray_image_array=gray_image_array)
    surf = SURF(image_array=image_array, gray_image_array=gray_image_array)
    orb  = ORB(image_array=image_array, gray_image_array=gray_image_array)
    #DECLARATIONS END

    #SHOW FEATURE POINTS BEGIN
    #sift.show_feature_points()
    #surf.show_feature_points()
    #orb.show_feature_points()
    #SHOW FEATURE POINTS END

    #SHOW FEATURE MATCHES BEGIN
    #sift.show_feature_matches()
    #surf.show_feature_matches()
    #orb.show_feature_matches()
    #SHOW FEATURE MATCHES END
    sift.stitch_images()  


