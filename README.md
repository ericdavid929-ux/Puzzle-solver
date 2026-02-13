# Puzzle-solver: Teaching My Computer how to do Puzzles

I decided to see if I could automate the most frustrating part of a puzzle: finding where the pieces go. This project uses Computer Vision (OpenCV and Python) to take images of scattered pieces and "snap" them into their correct positions on a reference board.

---

## Pipeline

To get this working, I built a pipeline that follows these steps:

* Isolation (Masking): First, I separate the pieces from the background. Depending on the lighting, I use different color spaces like LAB or RGB to make the pieces pop.
* Cropping: Once I have a clean binary mask, I find the contours (outlines) of every piece and crop them into individual images.
* Morphology: If two pieces are touching, I use Erosion and Dilation to "shrink" them until they separate, then "grow" them back to their original size.
* Brain Work (SIFT): I use the SIFT algorithm to find unique features on each piece. It’s like looking for tiny landmarks that match the main picture.
* The Final Snap: Using RANSAC and Homography, the code calculates the exact rotation and position needed to "warp" the piece onto the final canvas.

---

## The Results

### Peppa Pig (The Easy One)
This was the most successful run because the colors are flat and the pieces were already separated.
* Success Rate: 100%—every piece found its home.
* The Challenge: One piece had almost no features, but after some fine-tuning of the SIFT parameters, I got it to work.

### The City (The Hardest Ones)
Real photos are much harder because of the shadows and complex textures.
* City Blue: I managed to correctly place 38 out of 48 pieces. Some "boring" pieces (like plain pavement) just didn't have enough detail for the computer to recognize.
* City Black: This one had tricky lighting. The uneven brightness caused some pieces to be cropped poorly, but I still got 38 of them onto the board.

---

## Lessons Learned & Future Tweaks

* Lighting is very important: In the future, I’d use Adaptive Thresholding to handle shadows better, especially for the "City Black" set.
* Hole Filling: Some pieces ended up with "holes" in their masks; a quick filling method would make the final puzzle look much smoother.
* SIFT Tuning: I think I can get a 100% score on the city puzzles by tweaking the octave parameters or adding more detection methods.

---
By: David Cristian Eric
