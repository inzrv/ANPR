# Automatic number-plate recognition

This is part of the automatic number-plate recognition program. I am currently developing
number-plate detection stage.

Currently, neural networks or machine learning are not used here. Number-plates are detected using classic computer vision algorithms.

## Algorithm description

As input, the algorithm expects an image of size 512x512, which is stored in ```cv::Mat```. To prepare your image, you can use the ```prepare()``` function.

<img src="https://github.com/inzrv/ANPR/blob/main/examples/source.png" width="400"/> 

*Source image example*

In the second step, it finds the edges on the image using ```cv::Canny()```. At the same time, the algorithm uses dilation to keep the outline of the frame connected.

<p float="left">
  <img src="https://github.com/inzrv/ANPR/blob/main/examples/edges.png" width="400" />
  <img src="https://github.com/inzrv/ANPR/blob/main/examples/two_plates.png" width="400" /> 
</p>

*Found edges in the image (left). Number-plate frame after dilatation (right)*

The found edges are used for contour detection. For clarity, each contour is painted in its own color.

<img src="https://github.com/inzrv/ANPR/blob/main/examples/contours.png" width="400"/> 

*Contours in the image*

Now the algorithm approximates each contour and removes too short and too long of them.

<img src="https://github.com/inzrv/ANPR/blob/main/examples/approx_contours.png" width="400"/> 

*Approximate contours*

Delete the contours that don't look like rectangles. This is an interesting part of the algorithm, I think it can be implemented in different approaches. After that, there will be several polygons. Inside each of them may be a number-plate, so we must evaluate the probability of this for each polygon. For now, we'll just choose the largest in terms of area.

<p float="left">
  <img src="https://github.com/inzrv/ANPR/blob/main/examples/good_contours.png" width="400" />
  <img src="https://github.com/inzrv/ANPR/blob/main/examples/plate.png" width="400" /> 
</p>








