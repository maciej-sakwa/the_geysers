# The Geysers

## 1. Overview
This study focuses on analysis of seismic activity in The Geysers region in California, USA. 

In this research activity we wanted to study the presence of the earthquaqe frequency in the area and examine their spatial and time dependence. 
Our hypothesis was that these earthquakes are triggered by a composition of processes (or factors) and that their spatial and time density depends on the impacts of these factors. In this activity we try to identify those factors by means of unsupervised learning methods.

The full publication can be found at
[this link](https://ieeexplore.ieee.org/abstract/document/10418105).

## 2. The Geysers dataset

### 2.1. Geysers EGS

The Geysers geothermal power plant is currently the largest single geothermal power installation in the world, with a total installed capacity of 725 MW as of 2021. The power plant generates electricity from 22 geothermal power plants, which are spread across 45 square miles of the geothermal field. Recent studies have utilized a range of remote sensing techniques to better understand the subsurface structure and dynamics of the Enhanced Geothermal System (EGS).

<img src="img\area_california.png" alt="Geysers map" width="600"/>

The Geysers EGS is characterized by a complex network of faults and fractures that facilitates the circulation of hot fluids through the subsurface and accommodates via fault slipping the stress increase induced by the industrial processes connected to the energy production. Geologically, the geothermal
field is bordered by two major regional faults: Macaama and Collayomi.

Below the spatial density of the earthquakes in the area is plotted together
with the location of water injection wells and seismic stations.

<img src="img\geo_plot_log10_faults.png" alt="Events map" width="600"/>

Map of The Geysers EGS, California, US. Earthquake density over
the analyzed decade is displayed together with the position of the injection
wells (stars), seismic stations (squares), and faults (dashed lines).
other, 

### 2.2. Paper objective

Our work focuses on the analysis of an earthquake catalog recorded at The Geysers EGS. The dataset has been provided by the Northern California Earthquake Data Center (NCEDC). This archive is a joint project of the Berkley Seismology Laboratory (BSL) and the US Geological Survey (USGS) and it allows for specific queries on the entire dataset related to the geographical location, depth, event time, and magnitude of the recorded earthquakes. 

For the purpose of this work, we selected earthquakes occurring between January 2006 and June 2016 (126 months) in the region between 38.7° and 38.9° North, and between -122.95° and -122.65° West, that encloses all injection wells. In total, our dataset is composed of 421,344 earthquakes and includes seismicity up to about 10 km from the wells. The analyzed period includes about two cycles of water injection, during which the largest earthquake rate ever was detected. 



## 3. Code base structure

The data preprocessing is as follows:
1.   The control volume is divided into unit volumes with size of 0.25°x0.25°x 0.25 km (long-lat-depth) each, creating a grid of individual  cells that contain a number of seismic events each.
2. For each month of the observations we sum the number of events occurring in each cell in that time frame forming a spatial density dataset with a one-month time-step. 
3. By extracting the data of a single cell and arranging them in chronological order, we create an earthquake Density Time History (DTH) of each unit volume. DTH of a cell is the monthly rate of the earthquake that nucleates in that cell.

This extraction and ordering procedure has been performed for all
the cells in the analyzed dataset, resulting in a total of 5,254
DTH series. 

The entire pipeline is summarized by a block scheme below:

<img src="img\geysers_workflow.png" alt="Workflow" width="300"/>


The main code of the study has been divided into following files:

1. `src.clustering_module.py`: Hierarchical Agglomerative Clustering based on the time series (Density Time History) similarity of the control volume nodes.

2. `src.data_preparation.py`: Contains the data preprocessing pipeline with auxiliary functions.

3. `src.plotting_geo.py`: Contains the most important plotting functions.


## 4. Results

Based on the defined DTH the nodes have been clustered with HAC based on euclidean distance metric between the defined temporal vectors. According to a silhouette coefficient study, the optimal number of clusters to describe the studied earthquake catalogue is 5. However, due to low node count, cluster 1 and 2 have been rejected from further analysis. The studied clusters 3, 4, and 5 have been renamed as domain C, B, and A respectively. 

Their average DTH can be seen below:

<img src="img\DTH_ABC.png" alt="dths" width="400"/>

The clustering resulted in 3 domains with concentric behaviour as seen below:

<img src="img\eq_density_ABC.png" alt="domains" width="900"/>

Apart from geographic simillarity of the group members additional chacarceristics can be defined. 

### 4.1 Discussion

A discussion of the paper findigs with detailed description of the domains' characteristics can be found and accessed via [this link](https://ieeexplore.ieee.org/abstract/document/10418105). 


## 5. Credits 

Data processing and visualization: Maciej Sakwa

This research activity has been conducted in cooperation with dr Mauro Palo (Universita degli Studi di Napoli Federico II) and prof Emanuele Carlo Giovanni Ogliari (Politecnico di Milano) who both provided necessary scientific background and feedback on the development of the work.
