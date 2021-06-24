# PHT Metadata Examples
This folder contains some examples to illustrate the functionality of the Metadata in our PHT implementation.
## sampleStationMetadata
An exmeplary description of a Station in RDF (TURTLE Serialization), following the Schema. It describes a Station with one execution environment for executing trains and a single Dataset in the form of a json file. 
Note that the metadata schema was first developed implementation agnostic, so it supports for example describing multiple execution environments, while such concepts do not exist in our implementation.
## samplequery/ samplequery_result
An exemplary query to the TripleStore in SPARQL. This query requests all Logs over the CPU usage of a Train instance identified by a URL and corrosponding timestamps.
## screenshot_dashboard
A screenshot of the Dashboard. Different Panels are displaying different information such as the current station or the current state.
Under the hood, the dashboard utilizes queries such as the one shown above to retrieve the data and processes the results into visualizations.

