# Data Prep Steps

## Getting the data

Step to get the data are available [here](https://aws.amazon.com/public-data-sets/spacenet/)

You can also use the bash script `get_data <data-dir>` to get the entire data from aws and place it in `dir-name`

## Test to see if you are able to read the geojson files 

`
cd spacement/utilities/python/
`

`
./read3band.py
`

The response should be 

`251994
`

## Test to see if the Bounding Boxes are being generated correctly
`cd spacement/utilities/python/`

`overlay.py`

You should see the image below. THe black bounding boxes are used for training. 

<p align="center">
<img src="./utilities/python/Screenshot-of-bb-overlay.png" alt="Results" width="600px">
</p>





 
