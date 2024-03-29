# WCADS (WebCam Autonomous Driving System)
## introduction
it uses multiple webcams to navegate a preloaded or new enviroment using visual referecenses(recognisable lines, aruco markers, etc), it's kinda like a worse version of slam that can run a lot faster on much older hardwere with worse cameras

## what you need to use it
A robot that can move, a mini computer/laptop, at least 1 web cam(works better with 2) they dont need to have great resolution just good autofocus, a microcontroller that can recive movement comands in angles of rotation and meters to move in X and Y, so it needs its own way of doing dead reckoning to find how much it moved, i like to do this with encoders on each wheel and a simple simulation inside an arduino to aproximate location between WCADS updates.

## What currently works
- line detection
- aruco marker detection
- flor warping
