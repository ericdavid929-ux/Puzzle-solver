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
    image_path = os.path.join(project_root, 'data', 'city_black', 'city.jpg')
    piece1_path = os.path.join(project_root, 'data', 'city_black', 'pieces1.jpg')
    piece2_path = os.path.join(project_root, 'data', 'city_black', 'pieces2.jpg')
    piece3_path = os.path.join(project_root, 'data', 'city_black', 'pieces3.jpg')



    # Read The images 
    image = cv2.imread(image_path)
    piece1 = cv2.imread(piece1_path)
    piece2 = cv2.imread(piece2_path)
    piece3 = cv2.imread(piece3_path)


    # Safety check:
    if image is None or piece1 is None or piece2 is None or piece3 is None:
        print(f"Error: Could not find images at {image_path}")
    else:
        print("Images loaded successfully!")


    lab_piece1 = cv2.cvtColor(piece1, cv2.COLOR_BGR2LAB)
    L_piece1, A_piece1, B_piece1 = cv2.split(lab_piece1)

    lab_piece2 = cv2.cvtColor(piece2, cv2.COLOR_BGR2LAB)
    L_piece2, A_piece2, B_piece2 = cv2.split(lab_piece2)

    lab_piece3 = cv2.cvtColor(piece3, cv2.COLOR_BGR2LAB)
    L_piece3, A_piece3, B_piece3 = cv2.split(lab_piece3)


    _, binary_mask_piece1 = cv2.threshold(L_piece1, 128, 255, cv2.THRESH_BINARY)
    _, binary_mask_piece2 = cv2.threshold(L_piece2, 128, 255, cv2.THRESH_BINARY)
    _, binary_mask_piece3 = cv2.threshold(L_piece3, 128, 255, cv2.THRESH_BINARY)


    cv2.imshow('Mask1', binary_mask_piece1)
    cv2.waitKey(0)
    cv2.imshow('Mask2', binary_mask_piece2)
    cv2.waitKey(0)
    cv2.imshow('Mask3', binary_mask_piece3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    pieces1,masked_pieces1 = find_puzzle_pieces_with_masks(piece1,binary_mask_piece1,470)
    pieces2,masked_pieces2 = find_puzzle_pieces_with_masks(piece2,binary_mask_piece2,400)
    pieces3,masked_pieces3 = find_puzzle_pieces_with_masks(piece3,binary_mask_piece3,450)


    pieces=[]
    masked_pieces=[]

    pieces.extend(pieces1)
    pieces.extend(pieces2)
    pieces.extend(pieces3)

    masked_pieces.extend(masked_pieces1)
    masked_pieces.extend(masked_pieces2)
    masked_pieces.extend(masked_pieces3)




    # Create Puzzle Canvas
    puzzle_canvas = Lower_Transpernacy_image(image)


    # Find matches with SIFT detector
    sift = cv2.SIFT_create()
    h,w,_ = image.shape
    large_keypoints, large_descriptors = sift.detectAndCompute(image, None)

    for k in range(0,len(pieces)):
        # N.B. We manually extract the problematic pieces in order to compute the final puzzled assembled image
        if k in [14,33,37,38]:
            continue
        
        region_keypoints,region_descriptors,matches = find_matches(pieces[k],large_keypoints,large_descriptors,sift)
        
        matched_image = cv2.drawMatches(pieces[k], region_keypoints, image, large_keypoints, matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Matches", matched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        
        good_matches=filtered_matches_criteria(matches,4,0)

        matched_image = cv2.drawMatches(pieces[k], region_keypoints, image, large_keypoints, good_matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Matches", matched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        H = homography_computation(region_keypoints,large_keypoints,good_matches, 3.0)
        
        puzzle_canvas = pieces_assembly(pieces[k],masked_pieces[k],H,w,h,puzzle_canvas)

        cv2.imshow("Aligned Puzzle Piece", puzzle_canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()





    cv2.imshow('final', puzzle_canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()








