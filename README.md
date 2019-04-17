# Vehicle counting and speed measurement based on Video processing

Before going to the counting process of the vehicle and the vehicle speed measurement, in advance, we have to determine the area of ROI (Region of Interest). ROI is the area that became the observation area of moving vehicle. In the system, the ROI area is surrounded by two horizontal line and two vertical line. This is the most suitable area where car detection is mostly efficient. We need to pre-determine this area before processing Once ROI area is defined, then the vehicle counting and measuring vehicle speed followed by the detection of vehicle.
Figure 3.3 shows that above the counting line there is condition . As for below the counting line, there is a condition. The condition A is the Region of Interest (ROI) after passing the counting line. The logic used in this experiment is when a vehicle is in the state B and then passes the counting line, the system will calculate and measure the speed of vehicles. After the object passes the counting line and in condition A, the system stops the vehicle detection as well as the calculation and measurement of vehicle’s speed.

![roi](https://user-images.githubusercontent.com/29371886/56314362-adb70a00-6176-11e9-8456-484dbfa4b812.JPG)
                     Fig: Region Of Interest (ROI)
