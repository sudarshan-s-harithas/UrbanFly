# UrbanFly: Uncertainty-Aware Planning for Navigation Amongst High-Rises with Monocular Visual-Inertial SLAM Maps

UrbanFly: an uncertainty-aware realtime planning framework for quadrotor navigation in urban high-rise environments. A core aspect of UrbanFly is its ability to robustly plan directly on the sparse point clouds generated by a Monocular Visual Inertial SLAM (VINS) backend. Through UrbanFly we present two trajectory optimizers, The first optimizer uses gradient-free cross-entropy method to compute trajectories that minimize collision probability and smoothness cost. Our second optimizer is a simplified version of the first and uses a sequential convex programming optimizer initialized based on probabilistic safety estimates on a set of randomly drawn trajectories.Empowered by the algorithmic innovation, UrbanFly outperforms competing baselines in metrics such as collision rate, trajectory length, etc., on a high fidelity AirSim simulator augmented with synthetic and real-world dataset scenes


![](https://github.com/sudarshan-s-harithas/UrbanFly/blob/main/UrbanFlyCEMPlanner/Images/BlockDiagram.png)

#### Preprint: https://arxiv.org/pdf/2204.00865.pdf
#### YouTube: [Link](https://www.youtube.com/watch?v=ZmxUB3cMK4U)
### **Note: This code repository is under developent** 


