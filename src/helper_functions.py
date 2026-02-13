import numpy as np
import cv2 


# DEFINE SOME HELPER FUNCTIONS
def find_puzzle_pieces_with_masks(image,binary_mask,length):
    """
    This function takes as input an image (img) and its binary thresholded version(mask), both containing the puzzle pieces we want to segment 
    and it returns two lists (of images) containing respectivley the cropped puzzle pieces and their musk

    In order to find the picese, we firstly find the contrours (on the mask) and then we filter them (by setting the appropriate bounding box length)
    After doing that we bound the contours inside the boxes and we crop the corresponding piece
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for contour in contours:
        if cv2.boundingRect(contour)[2] > length and cv2.boundingRect(contour)[3] > length:
            filtered_contours.append(contour)

    
    masked_pieces = []
    pieces = []

    for contour in filtered_contours:
        # Get bounding box for each piece
        x, y, w, h = cv2.boundingRect(contour)
        
        # Crop the piece musk from the original image and store it
        masked_piece = binary_mask[y:y+h, x:x+w]
        masked_pieces.append(masked_piece)
    
        # Crop the piece from the original image and store it
        piece = image[y:y+h, x:x+w]
        pieces.append(piece)
    
        # Show the cropped piece and mask
        cv2.imshow('Intensity-based Cropped Piece', piece)
        cv2.waitKey(0)

        cv2.imshow('masked Cropped Piece', masked_piece)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    return pieces, masked_pieces


def Lower_Transpernacy_image(image):
    '''
    This function takes as input an image and outputs a brigther version of it
    N.B. it will be helpfull, as we will match the puzzle pieces on top of this brighter reference imgage to better observe 
    the process of puzzle assembly
    '''

    # Convert the image to BGRA (Blue, Green, Red, Alpha) format by adding an alpha channel
    image_with_alpha = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Adjust the alpha channel 
    alpha = image_with_alpha[:, :, 3]  
    alpha = alpha * 0.12  
    image_with_alpha[:, :, 3] = alpha  

    # Create a solid (white) background 
    background = np.ones_like(image_with_alpha, dtype=np.uint8) * 255 

    # Blend the image with the white background using the alpha channel
    # For R, G, B channels
    for c in range(0, 3):  
        image_with_alpha[:, :, c] = alpha * image_with_alpha[:, :, c] / 255 + (1 - alpha / 255) * background[:, :, c]

    # Convert back to RGB 
    final_image = cv2.cvtColor(image_with_alpha, cv2.COLOR_BGRA2BGR)

    # Display 
    cv2.imshow('Blended Image with Transparency', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return final_image


def find_matches(piece,large_keypoints,large_descriptors,sift):
    '''
    This function takes as input the segmented piece image, the keypoints and descriptors of the reference image and the sift 
    operator applied to the same reference image.
    It finds keypoints and descriptors of the segmented piece image, in order to use them to find the matches.
    Finally it returns the keypoints and descpritors of the cropped piece image and the list of mathces sorted
    '''
    region_keypoints, region_descriptors = sift.detectAndCompute(piece, None)

    # Use BFMatcher to find matches
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(region_descriptors, large_descriptors)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    return region_keypoints,region_descriptors,matches


def filtered_matches_criteria(matches,param,start):
    '''
    This function receives the matches and two values param and start and filters them. 
    It first find the minimum distance, and then retains the matches that satisfy the condition match.distance <= minimum_distance*param.
    In the case if minimum_distance is exactly zero we re-initialize it with the value start (as we don't want to lose good candidate matches).
    It finally returns a list with the filtered matches

    N.B. the matches are ordered by distance
    '''
    good_matches = []
    minimum_distance = matches[0].distance
    if minimum_distance == 0:
        minimum_distance += start
    for match in matches:
        if match.distance <= minimum_distance*param:
            good_matches.append(match)

    return good_matches

def homography_computation(region_keypoints,large_keypoints,good_matches, Ransac_Th):
    '''
    This fuction takes the keypoints of the reference image and the cropped piece, the list of filtered matches and the threshold value of RANSAC.
    It caluculated the spaical coordinates of the matches, in order to compute the homography matrix and returns it
    '''
    #spatial coordinated in cropped piece image
    src_pts = np.float32([region_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #spatial coordinated in reference image
    dst_pts = np.float32([large_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, Ransac_Th)
    
    return H

def pieces_assembly(piece,masked_piece,H,w,h,puzzle_canvas):
    '''
    This function takes the segmented piece and its mask, the homography matrix, the dimensions of the image and the background
    image (the brigther version of the reference image)
    It outputs the background image with the cropped piece in the correct position
    '''
    
    # black image with cropped puzzle piece(with background) on top assigned to the correct postion
    warped_piece = cv2.warpPerspective(piece, H, (w, h)) 
    # black image with masked puzzle piece on top assigned to the correct postion
    warped_mask = cv2.warpPerspective(masked_piece, H, (w, h))

    # inverse mask
    mask_inv = cv2.bitwise_not(warped_mask)
    #  Keep regions of the backgroung not covered by the segmented piece
    canvas_minus = cv2.bitwise_and(puzzle_canvas, puzzle_canvas, mask=mask_inv)  
    # Keep only the cropped piece's region
    piece_region = cv2.bitwise_and(warped_piece, warped_piece, mask=warped_mask)  

    #add the previous two to obtain the image with the correct puzzle postion
    final_canvas = cv2.add(canvas_minus, piece_region)
    
    return final_canvas
