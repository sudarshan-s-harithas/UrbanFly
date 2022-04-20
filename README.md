# UrbanFly: Uncertainty-Aware Planning for Navigation Amongst High-Rises with Monocular Visual-Inertial SLAM Maps

UrbanFly: an uncertainty-aware realtime planning framework for quadrotor navigation in urban high-rise environments. A core aspect of UrbanFly is its ability to robustly plan directly on the sparse point clouds generated by a Monocular Visual Inertial SLAM (VINS) backend. Through UrbanFly we present two trajectory optimizers, The first optimizer uses gradient-free cross-entropy method to compute trajectories that minimize collision probability and smoothness cost. Our second optimizer is a simplified version of the first and uses a sequential convex programming optimizer initialized based on probabilistic safety estimates on a set of randomly drawn trajectories.Empowered by the algorithmic innovation, UrbanFly outperforms competing baselines in metrics such as collision rate, trajectory length, etc., on a high fidelity AirSim simulator augmented with synthetic and real-world dataset scenes


![](https://github.com/sudarshan-s-harithas/UrbanFly/blob/main/UrbanFlyCEMPlanner/Images/BlockDiagram.png)

#### Preprint: https://arxiv.org/pdf/2204.00865.pdf
#### YouTube: [Link](https://www.youtube.com/watch?v=ZmxUB3cMK4U)
### **Note: This code repository is under developent** 


## Setup:

The planner evaluations were done on an 8-Core Intel Core i7-10870H processor, with 16GB RAM and 500GB HDD, running Ubunut 18.04 and ros melodic. we do recommend using powerfull setup.

A simulation server with NVIDIA1070 graphics card, AMD Ryzen 7 3800x 8-core processor× 16 CPU, and 64GB RAM and 1TB HDD was setup that hosted the simulation environments within the Unreal Engine. 

### Pre-requisites

[ROS Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu) <br />
[Unreal](https://docs.unrealengine.com/4.26/en-US/Basics/InstallingUnrealEngine/) <br />
[AirSim ROS Wrapper](https://microsoft.github.io/AirSim/airsim_ros_pkgs/) <br />
[PlaneRCNN](https://github.com/NVlabs/planercnn) <br />
[farneback3d](https://github.com/theHamsta/farneback3d) <br />


## Build


### Start the Simulation 

The Unreal Engine with the drone and the simulation environment would run on a remote server and the mapping and planning modules would be implemented on the local computer. Start the simulation and run the below command to connect Airsim to the remote host. Also do-not forget to configure ROS Master  slave . 
```
roslaunch airsim_ros_pkgs airsim_node.launch  host:=<ENTER_IP>
```
### State Estimation and Mapping

### CEM Planner 

Once the VINS is initilized and the planar map of the environment can be observed we are all set to start the CEM planner. Use the command given below to start the planner. 

```
rosrun UrbanFlyCEMPlanner CEMPlanner
```

### Implementation

![](https://github.com/sudarshan-s-harithas/UrbanFly/blob/main/UrbanFlyCEMPlanner/Images/Simulation.gif)

