# IMPORT PACKAGES
import numpy as np
import cv2 
import os

from helper_functions import find_puzzle_pieces_with_masks, Lower_Transpernacy_image, find_matches, filtered_matches_criteria, homography_computation, pieces_assembly

if __name__ == "__main__":
    # Get the directory where THIS script (is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) 

    # Construct the paths 
    image_path = os.path.join(project_root, 'data', 'peppa', 'peppa.png')
    pz_path = os.path.join(project_root, 'data', 'peppa', 'pz.png')

    # Load images
    image = cv2.imread(image_path)
    pz = cv2.imread(pz_path)

    # Safety check:
    if image is None or pz is None:
        print(f"Error: Could not find images at {image_path}")
    else:
        print("Images loaded successfully!")

    # Apply segmentation
    _,pz_binary_mask = cv2.threshold(pz[:,:,0], 254, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow('Mask', pz_binary_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pieces,masked_pieces = find_puzzle_pieces_with_masks(pz,pz_binary_mask,50)







    # Create Puzzle Canvas
    puzzle_canvas = Lower_Transpernacy_image(image)


    # Find matches with SIFT detector
    sift = cv2.SIFT_create()
    h,w,_ = image.shape

    large_keypoints, large_descriptors = sift.detectAndCompute(image, None)

    for k in range(0,len(pieces)):
        region_keypoints,region_descriptors,matches = find_matches(pieces[k],large_keypoints,large_descriptors,sift)
        
        matched_image = cv2.drawMatches(pieces[k], region_keypoints, image, large_keypoints, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Matches", matched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        
        good_matches=filtered_matches_criteria(matches,5,40)

        matched_image = cv2.drawMatches(pieces[k], region_keypoints, image, large_keypoints, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Matches", matched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        H = homography_computation(region_keypoints,large_keypoints,good_matches, 4.0)
        
        puzzle_canvas = pieces_assembly(pieces[k],masked_pieces[k],H,w,h,puzzle_canvas)

        cv2.imshow("Aligned Puzzle Piece", puzzle_canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    cv2.imshow('final', puzzle_canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
