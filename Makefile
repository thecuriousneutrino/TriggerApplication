#CUDA_HOME=/usr/local/cuda
CUDA_HOME=/usr/local/cuda-8.0
CUDAINC = -I$(CUDA_HOME)/include 
CUDALIB = -L$(CUDA_HOME)/lib64 -lcudart -lcuda -lcudadevrt

#NVCCFLAGS       := -lineinfo -arch=sm_20 --ptxas-options=-v --use_fast_math
NVCCFLAGS       := -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math 

ToolDAQPath=ToolDAQ
ZMQLib= -L $(ToolDAQPath)/zeromq-4.0.7/lib -lzmq 
ZMQInclude= -I $(ToolDAQPath)/zeromq-4.0.7/include/ 

BoostLib= -L $(ToolDAQPath)/boost_1_66_0/install/lib -lboost_date_time -lboost_serialization -lboost_iostreams
BoostInclude= -I $(ToolDAQPath)/boost_1_66_0/install/include

DataModelInclude = $(RootInclude) $(WCSimInclude)
DataModelLib =  $(RootLib) $(WCSimLib)

MyToolsInclude = 
MyToolsLib = 

MyToolsIncludeGPU = $(MyToolsInclude) $(CUDAINC)
MyToolsLibGPU = $(MyToolsLib) $(CUDALIB)

RootInclude := -I$(shell root-config --incdir)
RootLib     := $(shell root-config --libs)

WCSimInclude = -I$(WCSIMDIR)/include
WCSimLib     = -L$(WCSIMDIR) -lWCSimRoot

ifdef BONSAIDIR
	# if directory BONSAIDIR exists:
	BonsaiInclude = -I$(BONSAIDIR)/bonsai
	BonsaiLib = -L$(BONSAIDIR) -lWCSimBonsai
endif

ifdef EBONSAIDIR
    EBonsaiInclude = -I$(EBONSAIDIR)
    EBonsaiLib = -L$(EBONSAIDIR) -lWCSimEBonsai
endif

CXXFLAGS = -g -std=c++11 -Wpedantic

all: lib/libStore.so lib/libLogging.so lib/libDataModel.so include/Tool.h lib/libMyTools.so lib/libServiceDiscovery.so lib/libToolChain.so main RemoteControl  NodeDaemon

GPU: lib/libStore.so lib/libLogging.so lib/libDataModel.so include/Tool.h lib/libMyToolsGPU.so lib/libServiceDiscovery.so lib/libToolChain.so mainGPU RemoteControl NodeDaemon

main: src/main.cpp | lib/libMyTools.so lib/libStore.so lib/libLogging.so lib/libToolChain.so lib/libDataModel.so lib/libServiceDiscovery.so
	@echo "\n*************** Making " $@ "****************"
	g++ $(CXXFLAGS) src/main.cpp -o main -I include -L lib -lStore -lMyTools -lToolChain -lDataModel -lLogging -lServiceDiscovery -lpthread $(DataModelInclude) $(DataModelLib) $(MyToolsInclude)  $(MyToolsLib) $(ZMQLib) $(ZMQInclude)  $(BoostLib) $(BoostInclude)

mainGPU: src/main.cpp UserTools/CUDA/GPU_link.o | lib/libMyToolsGPU.so lib/libStore.so lib/libLogging.so lib/libToolChain.so lib/libDataModel.so lib/libServiceDiscovery.so
	@echo "\n*************** Making " $@ "****************"
	g++ $(CXXFLAGS) src/main.cpp UserTools/CUDA/GPU_link.o -o main -I include -L lib -lStore -lMyToolsGPU  -lMyTools -lToolChain -lDataModel -lLogging -lServiceDiscovery -lpthread $(DataModelInclude) $(DataModelLib) $(MyToolsIncludeGPU)  $(MyToolsLibGPU) $(ZMQLib) $(ZMQInclude)  $(BoostLib) $(BoostInclude)


lib/libStore.so: $(ToolDAQPath)/ToolDAQFramework/src/Store/*
	cd $(ToolDAQPath)/ToolDAQFramework && make lib/libStore.so
	@echo -e "\n*************** Copying " $@ "****************"
	cp $(ToolDAQPath)/ToolDAQFramework/src/Store/*.h include/
	cp $(ToolDAQPath)/ToolDAQFramework/lib/libStore.so lib/
	#g++ $(CXXFLAGS) -fPIC -shared  -I include $(ToolDAQPath)/ToolDAQFramework/src/Store/*.cpp -o lib/libStore.so $(BoostLib) $(BoostInclude)


include/Tool.h:  $(ToolDAQPath)/ToolDAQFramework/src/Tool/Tool.h
	@echo -e "\n*************** Copying " $@ "****************"
	cp $(ToolDAQPath)/ToolDAQFramework/src/Tool/Tool.h include/


lib/libToolChain.so: $(ToolDAQPath)/ToolDAQFramework/src/ToolChain/* | lib/libLogging.so lib/libStore.so lib/libMyTools.so lib/libServiceDiscovery.so lib/libLogging.so lib/libDataModel.so
	@echo -e "\n*************** Making " $@ "****************"
	cp $(ToolDAQPath)/ToolDAQFramework/UserTools/{Factory,Logger,ServiceAdd}/*.h include/
	cp $(ToolDAQPath)/ToolDAQFramework/UserTools/Unity.h include/
	cp $(ToolDAQPath)/ToolDAQFramework/src/ToolChain/*.h include/
	g++ $(CXXFLAGS) -fPIC -shared $(ToolDAQPath)/ToolDAQFramework/src/ToolChain/ToolChain.cpp -I include -lpthread -L lib -lStore -lDataModel -lServiceDiscovery -lMyTools -lLogging -o lib/libToolChain.so $(DataModelInclude) $(DataModelLib) $(ZMQLib) $(ZMQInclude) $(MyToolsInclude)  $(BoostLib) $(BoostInclude)



clean: 
	@echo -e "\n*************** Cleaning up ****************"
	rm -f include/*.h
	rm -f lib/*.so
	rm -f main
	rm -f RemoteControl
	rm -f NodeDaemon
	rm -f UserTools/CUDA/*.o

lib/libDataModel.so: DataModel/* lib/libLogging.so | lib/libStore.so
	@echo -e "\n*************** Making " $@ "****************"
	cp DataModel/*.h include/
	g++ --version
	g++ $(CXXFLAGS) -fPIC -shared DataModel/*.cpp -I include -L lib -lStore  -lLogging  -o lib/libDataModel.so $(DataModelInclude) $(DataModelLib) $(ZMQLib) $(ZMQInclude)  $(BoostLib) $(BoostInclude)

lib/libMyTools.so: UserTools/*/* UserTools/* | include/Tool.h lib/libDataModel.so lib/libLogging.so lib/libStore.so
	@echo "\n*************** Making " $@ "****************"
	cp UserTools/*/*.h include/
	cp UserTools/Factory/*.h include/
	g++ $(CXXFLAGS) -fPIC -shared  UserTools/Factory/Factory.cpp -I include -L lib -lStore -lDataModel -lLogging -o lib/libMyTools.so $(MyToolsInclude) $(MyToolsLib) $(DataModelInclude) $(DataModelLib) $(ZMQLib) $(ZMQInclude) $(BoostLib) $(BoostInclude) $(BonsaiLib) $(BonsaiInclude) $(EBonsaiLib) $(EBonsaiInclude)

lib/libMyToolsGPU.so: UserTools/*/* UserTools/* UserTools/CUDA/GPU_link.o | include/Tool.h lib/libDataModel.so lib/libLogging.so lib/libStore.so
	@echo "\n*************** Making " $@ "****************"
	cp UserTools/*/*.h include/
	cp UserTools/Factory/*.h include/
	g++ $(CXXFLAGS) fPIC -shared  UserTools/Factory/Factory.cpp  -DGPU UserTools/CUDA/CUDA_Unity.o -I include -L lib -lStore -lDataModel -lLogging -o lib/libMyToolsGPU.so $(MyToolsIncludeGPU) $(MyToolsLibGPU) $(DataModelInclude) $(DataModelLib) $(ZMQLib) $(ZMQInclude) $(BoostLib) $(BoostInclude)

RemoteControl:
	cd $(ToolDAQPath)/ToolDAQFramework/ && make RemoteControl
	@echo -e "\n*************** Copying " $@ "****************"
	cp $(ToolDAQPath)/ToolDAQFramework/RemoteControl ./

NodeDaemon: 
	cd $(ToolDAQPath)/ToolDAQFramework/ && make NodeDaemon
	@echo -e "\n*************** Copying " $@ "****************"
	cp $(ToolDAQPath)/ToolDAQFramework/NodeDaemon ./

lib/libServiceDiscovery.so: $(ToolDAQPath)/ToolDAQFramework/src/ServiceDiscovery/* | lib/libStore.so
	cd $(ToolDAQPath)/ToolDAQFramework && make lib/libServiceDiscovery.so
	@echo -e "\n*************** Copying " $@ "****************"
	cp $(ToolDAQPath)/ToolDAQFramework/src/ServiceDiscovery/ServiceDiscovery.h include/
	cp $(ToolDAQPath)/ToolDAQFramework/lib/libServiceDiscovery.so lib/
	#g++ $(CXXFLAGS) -shared -fPIC -I include $(ToolDAQPath)/ToolDAQFramework/src/ServiceDiscovery/ServiceDiscovery.cpp -o lib/libServiceDiscovery.so -L lib/ -lStore  $(ZMQInclude) $(ZMQLib) $(BoostLib) $(BoostInclude)

lib/libLogging.so:  $(ToolDAQPath)/ToolDAQFramework/src/Logging/* | lib/libStore.so
	cd $(ToolDAQPath)/ToolDAQFramework && make lib/libLogging.so
	@echo -e "\n*************** Copying " $@ "****************"
	cp $(ToolDAQPath)/ToolDAQFramework/src/Logging/Logging.h include/
	cp $(ToolDAQPath)/ToolDAQFramework/lib/libLogging.so lib/
	#g++ $(CXXFLAGS) -shared -fPIC -I include $(ToolDAQPath)/ToolDAQFramework/src/Logging/Logging.cpp -o lib/libLogging.so -L lib/ -lStore $(ZMQInclude) $(ZMQLib) $(BoostLib) $(BoostInclude)

update:
	@echo -e "\n*************** Updating ****************"
	cd $(ToolDAQPath)/ToolDAQFramework; git pull
	cd $(ToolDAQPath)/zeromq-4.0.7; git pull
	git pull

UserTools/CUDA/GPU_link.o:  UserTools/CUDA/*
	@echo "\n*************** Compiling & Linking " $@ "****************"
	cp UserTools/CUDA/*.h include/
	nvcc -c --shared -Xcompiler -fPIC -dlink UserTools/CUDA/CUDA_Unity.cu -o UserTools/CUDA/CUDA_Unity.o  -I UserTools/CUDA/  $(CUDAINC) $(NVCCFLAGS) $(CUDALIB)
	nvcc  -arch=sm_35 -dlink  -o UserTools/CUDA/GPU_link.o UserTools/CUDA/CUDA_Unity.o  $(CUDALIB) $(CUDAINC)
