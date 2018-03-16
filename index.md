## Abstract

Simulators are useful for developing algorithms
and systems for autonomous driving, but it is challenging to
model realistic multi-agent driving behavior. We study how
to leverage an online public traffic cam video stream to
extract data of driving behavior to use in an open-source
traffic simulator, FLUIDS. To tackle challenges like frameskip,
perspective, and low resolution, we implement a Traffic
Cam Pipeline (TCP). TCP leverages recent advances in deep
object detection and filtering to extract trajectories from the
video stream to corresponding locations in a bird’s eye view
traffic simulator. After collecting 2618 car and 1213 pedestrian
trajectories, we modify the simulator’s multi-agent planner to
reflect the learned behaviors in the real world intersection.
Specifically, we examine how to estimate parameters for the
simulator’s high-level behavioral logic and the generated motion
plans for pedestrians and cars through online traffic videos.