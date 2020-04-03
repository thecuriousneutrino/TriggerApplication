# TriggerApplication

Trigger Application is designed as a modular trigger software to run successive triggers on WCSim data. 

It's built from ToolDAQ Application [[1]](#myfootnote1) which is an open source general DAQ Application template built using the modular ToolDAQ Framework core [[2]](#myfootnote2) to give separation between core and implementation code.

****************************

# Concept

****************************

The main executable creates a ToolChain which is an object that holds Tools. Tools are added to the ToolChain and then the ToolChain can be told to Initialise Execute and Finalise each tool in the chain.

The ToolChain also holds a user-defined DataModel which each tool has access too and can read, update and modify. This is the method by which data is passed between Tools.

User Tools can be generated for use in the tool chain by including a Tool header. This can be done manually or by use of the newTool.sh script.

For more information consult the ToolDAQ doc.pdf

https://github.com/ToolDAQ/ToolDAQFramework/blob/master/ToolDAQ%20doc.pdf

****************************

# Tutorial

Note that there are `README.md` files in most folders. Check them out!

* [Key concepts](#key-concepts)
  * [Tool](#tool)
  * [Toolchain](#toolchain)
  * [DataModel](#datamodel)
* [Installation](#installation)
  * [Docker](#docker)
  * [From GitHub source](#from-github-source)
* [Running](#running)
  * [Creating your own trigger chain](#creating-your-own-trigger-chain)
  * [Creating your own trigger](#creating-your-own-trigger)

## Key concepts

### Tool

A tool is a class that does something. They are found in the `UserTools` directory. Examples includes
* `WCSimReader` -- Reads in WCSim files
* `nhits` -- Runs an nhits trigger
* `DataOut` -- Writes out WCSim-format files with only triggered digits

Each tool implements 3 methods
* `Initialise()` -- Is run once at the start. Read configuration, setup trees, etc. here.
* `Execute()` -- Is run once per event. Do the main work of the tool here.
* `Finalise()` -- Is run once at the end. Close & write files, `delete` memory, etc. here

### Toolchain

A toolchain is a collection of tools, along with configuration files. They are found in the `configfiles` directory. For example
* `WCSimReader` - `nhits` - `DataOut` -- Reads a WCSim file, triggers on it, and writes out a new WCSim-formatted file

Note that you can use the same tool multiple times, with different (or the same) configuration

Note that tools are run consecutively. i.e. if you have two tools (t1, t2) the following order will be used:
* `t1::Initialise()` - `t2::Initalise()` - `t1::Execute()` - `t2::Execute()` - `t1::Execute()` - `t2::Execute()` - ... - `t1::Finalise()` - `t2::Finalise()`

### DataModel

Tools cannot communicate directly with one another. They rely on passing data between each other using a transient data model. This is found in the `DataModel` folder

## Installation

### Docker

Docker is a platform independent container system which eases the headache of long installation and incompatibility problems. By creating a container form the image you will have a native Centos 7 terminal with all the prerequisites installed and the software guaranteed to work.

1) Install docker check either your platforms package manager or https://www.docker.com for the software
2) Get the latest container image `docker pull hkdaq/triggerapplication:latest`
3) Run an instance of the container which will have the trigger application and all dependencies installed `docker run --name=TriggerApplication -it hkdaq/triggerapplication:latest` Note: only run once or you will make multiple contianers

Once the container has started to run the software
1) `./main`
  * This runs an example toolchain with two versions of the `dummy` tool. It's essentially a Hello World tool

You're then free to install any applications in your container you wish for development

Notes: 
* To exit a container use `exit` 
* To restart a container use `docker start -i TriggerApplicaiton`
* To see current containers use `docker ps -a`
* To delete a container use `docker rm TriggerApplciaiton`
* When creating a container you can mount a folder from your native os with the `-v` run option e.g. `docker run --name=TriggerApplication -v local_folder_path:container_mount_path -it hkdaq/triggerapplication:latest`

### From GitHub source

* Clone from https://github.com/HKDAQ/TriggerApplication
  * Note the model used to commit to the main version of TriggerApplication is fork and pull request. So do fork if you need to!
* Make sure you have sourced WCSim i.e. that you have `$WCSIMDIR` set
  * Note that this will work with the current WCSim develop branch.
    * Versions of WCSim older than v1.8.0 will almost certainly not work. (`kTriggerNoTrig` added in v1.8.0; `WCSimRootOptions` added in v1.7.0)
  * Note that you also need ROOT setup (a WCSim prerequisite)
* (Optional) If you want to run the BONSAI tool, make sure you have sourced hk-BONSAI i.e. that you have `$BONSAIDIR` set
* (Optional) If you want to run the EnergeticBONSAI tool, make sure you have sourced energetic-bonsai i.e. that you have `$EBONSAIDIR` set
  * Note: you need to use the library branch at https://github.com/tdealtry/energetic-bonsai/tree/library (PR pending)
* Run `./GetToolDAQ.sh`
  * This gets and compiles the prerequisites: ToolDAQ, boost, and zmq
  * You can optionally install Root
  * `./GetToolDAQ.sh --help` for the flags to turn on/off each of the prerequisites

To check it has built successfully:
* `source Setup.sh`
* Check it runs with `./main`
  * This runs an example toolchain with two versions of the `dummy` tool. It's essentially a Hello World tool

#### GPU code

Some triggers have been developed to be run on CUDA-compatible GPUs.
If you want to use these (and you have a compatible system)
* `source Setup.sh`
* `make GPU`
* `./mainGPU`

## Running

1. Choose the toolchain you want to run
  * We use `WCSimReaderTest` as an example
  * See https://github.com/WCSim/WCSim for how to compile and run WCSim
    * Running `cd $WCSIMDIR; ./bin/Linux-g++/WCSim WCSim.mac` will create the expected output file for this tutorial.
2. Check the configuration files are doing what you want them to in `configfiles/WCSimReaderTest`
  * `ToolChainConfig` -- Sets up how many events to run on, what to do on errors, etc. You probably don't need to alter this
  * `ToolsConfig` -- Select which tool(s) you want to use, and the configuration file of each version of the tool
  * `WCSimReaderToolConfig` -- Options for the `WCSimReader` tool. Select the input file(s), number of events to loop over, and tool verbosity
    * Note the default WCSim input file is `$WCSIMDIR/wcsim.root`
  * `nhitsToolConfig` -- Select the trigger options (e.g. threshold), whether to apply it to ID or OD digits, and tool verbosity
  * `DataOutToolConfig` -- Select the output filename, whether to save multiple digits per PMT per trigger, the digit time offset, and tool verbosity.
    * Note the default output file is `triggered.root`
3. Run as `./main WCSimReaderTest`

## Creating your own trigger chain

1. `cd $ToolDAQapp/configfiles`
2. `./Create_run_config.sh TOOLCHAINNAME`
3. Write you configuration files in `$ToolDAQapp/configfiles/TOOLCHAINNAME`
4. `cd $ToolDAQapp`
  * Note you can setup your configuration files with absolute paths such that you don't need to `cd`
5. Run with `./main TOOLCHAINNAME`

## Creating your own trigger

1. `cd $ToolDAQapp/UserTools`
2. `./newTool.sh TOOLNAME`
   * Note the convention for triggers is to start TOOLNAME with a lower case. For other tools, start with an upper case
3. Write your trigger in the new class that has been created (`$ToolDAQapp/UserTools/TOOLNAME/TOOLNAME.{cpp,h}`files
   * Implement `Initalise()`
     * `m_variables.Get("verbose", verbose);` is an example of reading in configuration options
   * Implement `Execute()`
     * Triggers shouldn't read truth information (although you could implement truth cherry pickers...) so you should only use the following from the DataModel `m_data`
       * **Inputs**
         * `IDSamples` and `ODSamples` contain all the digit information i.e. charge, time, tubeID
         * `IDGeom` and `ODGeom` contain all the PMT information i.e tubeID, x, y, z

           * Note that this can be expanded to include e.g. PMT rotation

       * **Outputs**
         * Use `TriggerInfo::AddTrigger()` to save triggers in `IDTriggers` or `ODTriggers`
     * You can implement both CPU and CUDA-based GPU versions of your code
       * It is recommended to always have a CPU version of the code, since this allows anyone to use it; access to GPUs is not ubiquitous
       * Use the following to select the correct version of the code, and hide GPU code from systems that cannot compile it 
        ```
        #ifdef GPU
        // GPU code
        #else
        // CPU CODE
        #endif //GPU
        ```

   * Implement `Finalise()`
     * Remember to `delete` any memory you've allocated
   * Check other triggers for more information
      * `pass_all` is a very simple example
      * `nhits` is a relatively simple example. It has CPU and GPU versions
4. Use `make` and/or `makeGPU` to build it
5. Add your tool to a toolchain to test it with `./main TOOLCHAINNAME`
  * Or `./mainGPU TOOLCHAINNAME` for GPU code
6. Write the README: `$ToolDAQapp/UserTools/TOOLNAME/README.md`

****************************

Copyright (c) 2018 Hyper-k Collaboration

<a name="myfootnote1">[1]</a> Benjamin Richards. (2018, November 11). ToolDAQ Application v2.1.2 (Version V2.1.2). Zenodo. http://doi.org/10.5281/zenodo.1482772

<a name="myfootnote2">[2]</a> Benjamin Richards. (2018, November 11). ToolDAQ Framework v2.1.1 (Version V2.1.1). Zenodo. http://doi.org/10.5281/zenodo.1482767 
