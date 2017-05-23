# Advanced Lane Finding
---
### Project 4 writeup
---
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_stuff/chess_orig.png "Original Image"
[image2]: ./writeup_stuff/chess_undistorted.png "Undistorted Image"
[image3]: ./writeup_stuff/straight_lines.png.png "Distortion correction"
[image4]: ./writeup_stuff/color_binary.png.png "Color binary and combined threshold Image"
[image5]: ./writeup_stuff/pers1.png "Source and destination points for straight lanes"
[image6]: ./writeup_stuff/pers2.png "Original image and Perspective transform "
[image7]: ./writeup_stuff/pers3.png "Source and destination points for curved lanes"
[image8]: ./writeup_stuff/pers4.png "Original image and Perspective transform Image"
[image9]: ./writeup_stuff/binary_warped.png "Binary Warped"
[image10]: ./writeup_stuff/lane_lines.png "Lane lines"
[image11]: ./writeup_stuff/draw_lanes.png "Final output image"
[video1]: ./output_images/project_video_output.mp4 "Output video"
---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

- For capturing the camera matrix values and distortion coefficients, I referred the sample code written in [Camera Calibration - Github](https://github.com/udacity/CarND-Camera-Calibration).

The code for this step is contained in the third code cell of the IPython notebook located in "./P4.ipynb" 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Original Image][image1]

![Undistorted Image][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
The distortion correction coefficients calculated in code cell 3 are used to undistort the test images. This code can be observed in cell 4
```python
file_name = "test_images/straight_lines1.jpg"
image = cv2.imread(file_name)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
undistorted_img = undistort_image(image, objpoints, imgpoints)
if imshow_enable == 1:
    plot_images(image, undistorted_img, "Original Image", "Undistorted Image")
```

![Left: Original Image, Right: Undistorted Image][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In code cell 6, I have a routine called 'binary_image' which takes in undistorted image and returns color binary and combined binary images.

Initially, we convert images to HLS color space and keep the S channel of image. The rationale behind choosing S channel is that it gives lane markings better regardless of color of lane. 

For capturing the lane markings in vertical direction, Sobel gradient is applied with a threshold of 20 and 100. Finally, the S-channel binary and Sobel gradient binary image are stacked together to create combined binary which is used in our pipeline for processing.

Sample output from my function is shown below:
![Binary Image][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in the 4th code cell of the IPython notebook).  The `perspective_transform()` function takes as inputs an image (`img_src`). The `src` and `dst` points are calculated in `get_src_dst_wrap_points()` function.  I chose the hardcode the source and destination points in the following manner:

```python
    left_top = [585, 460]
    left_bottom = [255, 700]
    right_top = [700, 460]
    right_bottom = [1060, 700]
    src = np.array([ left_top, left_bottom, right_bottom, right_top], np.int32)

    offset = 50
    left_edge = left_bottom[0] + offset
    right_edge = right_bottom[0] - offset
    
    left_top = [left_edge, 0]
    left_bottom = [left_edge, left_bottom[1]]
    right_top = [right_edge, 0]
    right_bottom = [right_edge, left_bottom[1]]

    dst = np.array([ left_top, left_bottom, right_bottom, right_top], np.int32)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 305, 0        | 
| 255, 700      | 305, 700      |
| 1060, 700     | 1010, 700      |
| 700, 460      | 1010, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Source and destination points][image5]

![Transformed image][image6]

![Source and destination points][image7]

![Transformed image][image8]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The image from obtained from color thresholding is then passed through  perspective transform routine.
For finding lanes, I followed the approach mentioned in lessons and reformatted the code into `find_lane_lines()` function which takes binary warped image as input and `debug` value as bool for debugging purpose.

In this approach, we make use of fact that since in thresholded binary image, all pixels are 0 or 1, the most prominent peaks in histogram will likely be the place belonging to lane lines. 

![Binary Warped Lane image][image9]

Using such points as starting points, we use a sliding window to follow the lanes to top of frame. The result of such operation is shown below.

![Lane lines ][image10]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

For finding curvature, we use x and y points calculated in earlier step to fit a second order polynomial.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in code cell 11 in the function `draw_lanes()`.  Here is an example of my result on a test image:

![Final output image][image11]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  