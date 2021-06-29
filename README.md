# MIM
Methods of Information in Medicine


## DA software
Our used containers for the station software can be found here: https://hub.docker.com/u/smithpht



## Metadata
See [here](https://github.com/LaurenzNeumann/PHTMetadata) the complete Metadata Schema.
The Metadata schema provides a Standard to express infromation about different entities in the PHT architecture.
It is implemented in RDFS and contains constraints expressed in SHACL. Our implementation does not only provide infrastructure to automatically collect Metadata from the Stations but also a Dashboard for processing it and displaying it in user-friendly visualizations.
You can find examples in the Examples_Metadata folder.

# Usecase

We apply our PHT implementation to the use case "Pneumonia Detection" as a proof-of-concept approach. 
The data set is available on [Mendeley Dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2). 
We provide the example code both for single server simulation and also docker solution for running on our PHT infrastructure.

