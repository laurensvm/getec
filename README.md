# End-To-End Genre Prediction System

## Project Structure
The application consists of seven different sections. Each section is encapsulated and should function on itself. The project structure discussed proved to be very useful to keep everything modular and organised. We recommend using a similar structure for small machine learning projects. We briefly discuss each section.

### App
The first section is the App. This is the entry point of the application. We tell the application what we want to do (training, testing, downloading, processing) and it takes care of the full process based on our input parameters. The App can be seen as the general manager of the application, and holds a reference to all the other sections. Communication between sections is done through the App unless inappropriate.

### I/O Handler (File System Manager)
This section takes care of everything related to storing and reading files. It stores locations of all the files and folders that we need during the entire process. When running the application for the first time, this section creates a directory to store the songs in and a directory to save trained models in. It is also responsible for reading mp3 files and converting them to wav.

### Data Downloader
The data downloading section is responsible for downloading all the required songs and put them in the appropriate genre folder, with help of the file system manager. Here we also have a text file which contains id’s of all the songs we have already downloaded, so that we do not download duplicate songs.

### Pre-Processor
The third section is the pre-processor. In this section, everything related to transforming the audio samples is performed. The pre-processor selects multiple audio fragments from a song, creates spectrograms and reduces the spectrograms to a dense normalised matrix.

### Dataset
The dataset section is responsible for serialising the genre and the corresponding normalised matrices created in the pre-processor into bytes, and saving them in a database file. Upon retrieval of the file, before training or testing, it deserialises the objects and splits them into training data, validation data and testing data. Consequently, all of the data is shuffled and ready to use for the neural networks.

### Networks
The networks section contains the configuration and implementation for a standard neural network, a recurrent neural network, a convolutional neural network and a combination of recurrence and convolution. The networks can load themselves from the saved models directory, or train a new network.

### Visualiser
The final section is what we call the visualiser. This section is responsible for the visualisations that are produced from the results. During training, the network keeps track of various statistics. After training, these statistics are passed into the visualiser, which creates plots from them.
