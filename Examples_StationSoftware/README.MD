# Screenshots showing the interaction with our Software
## Central Service Dashboard (Train Requester)
The researcher requests a Train, and the Central Service manages the routing and communication with the stations. Finally, the researcher is able to inspect the results.

The researcher can observe the list of requested trains and their current status.
![Train Requester](./CS_Train_Request.JPG?raw=true)

Here, the researcher makes a new train request, chooses a Train from App Store, and selects the Station sequence to be visited. 
![Train Requester](./CS_Make_a_Request.JPG?raw=true)


## Station Software
The Station Admin manages settings and requested trains via the Station software.

A list of waiting Trains is displayed, and The Station Admin decides which Train to pull.
![Station Software](./Station_Software_Train_Poll.JPG?raw=true)

The Station Admin observes the pulled Trains; the pulled trains need to be decrypted for execution. The decryption process is done in the backend using the Station RSA key pair.
![Station Software](./Station_Software_Train_Pulled.JPG?raw=true)

The Station Admin adjusts the settings that the Train needs to run. The requirements are defined as a part of the Train metadata. These settings are provided to the created container via environment variables.
![Station Software](./Station_Software_Train_Interfaces.JPG?raw=true)

The Station Admin observes the latest status of the executions. The logs and changes made by the execution can be inspected.
![Station Software](./Station_Software_Train_Executed.JPG?raw=true)

As shown in the following screenshots, the results can be inspected via the Station software GUI.
Three possible changes are considered, `Modified`, `Added`, `Changed`. In the "Compare" view, the station admin can audit what has exactly changed in the contents of a particular file. However, only the parts of the file contents that are relevant to the current station are visible, and the information from other stations is hidden. After the audition step, the Station Admin can commit the container. This step encrypts the results and prepares the Train to return to the center.
![Station Software](./resultSetMasking.JPG?raw=true)
![Station Software](./resultSetMaskingExtended.JPG?raw=true)

In the last step, the Station Admin pushes back the Train to the center for further process.
![Station Software](./Station_Software_Train_Push.JPG?raw=true)
