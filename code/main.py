from sift import SIFT
from orb import ORB
from surf import SURF
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, destroyAllWindows


DESCRIPTION = """
IMAGE STITCHING WITH SIFT, SURF AND ORB FEATURE DETECTORS
    SIFT:
        to show feature points  type 1.1
        to show feature matches type 1.2
        to show panorama        type 1.3
    
    SURF:
        to show feature points  type 2.1
        to show feature matches type 2.2
        to show panorama        type 2.3

    ORB:
        to show feature points  type 3.1
        to show feature matches type 3.2
        to show panorama        type 3.3
    
    TO SEE DESCRIPTION          type desc
    TO CLOSE WINDOWS            type clean
    TO FINISH                   type X
"""

if __name__ == "__main__":
    image_array = []
    gray_image_array = []
    #TODO: This code must be changed based on max and min image id.
    #TODO: Add a console UI to control the sw
    for ix in range(4, 10):
        image_file_name = "dataset/cyl_image" + ( "0" + str(ix)  if ix < 10 else str(ix)) + ".png"
        image_array.append(imread(image_file_name))  
        gray_image_array.append(cvtColor(image_array[-1], COLOR_BGR2GRAY))      
    
    #DECLARATIONS BEGIN
    sift = SIFT(image_array=image_array, gray_image_array=gray_image_array)
    surf = SURF(image_array=image_array, gray_image_array=gray_image_array)
    orb  = ORB(image_array=image_array, gray_image_array=gray_image_array)
    #DECLARATIONS END

    print(DESCRIPTION)
    while True:
        user_input = input("type:")

        if user_input == "X":
            print("Bye Bye")
            break
        
        elif user_input == "clean":
            destroyAllWindows()

        elif user_input == "desc":
            print(DESCRIPTION)

        elif user_input.startswith("1."):#SIFT
            if user_input == "1.1": sift.show_feature_points()
            elif user_input == "1.2": sift.show_feature_matches()
            elif user_input == "1.3": sift.stitch_images()
        elif user_input.startswith("2."):#SURF
            if user_input == "2.1": surf.show_feature_points()
            elif user_input == "2.2": surf.show_feature_matches()
            elif user_input == "2.3": surf.stitch_images()
        elif user_input.startswith("3."):#ORB
            if user_input == "3.1": orb.show_feature_points()
            elif user_input == "3.2": orb.show_feature_matches()
            elif user_input == "3.3": orb.stitch_images()

        else:print("Wrong Type, Read Description Shown Above")

