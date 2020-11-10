CUDA_HOME=$(CUDADIR)
#CUDA_HOME=/usr/local/cuda-8.0
CUDAINC = -I$(CUDA_HOME)/include 
CUDALIB = -L$(CUDA_HOME)/lib64 -lcudart -lcuda -lcudadevrt

#NVCCFLAGS       := -lineinfo -arch=sm_20 --ptxas-options=-v --use_fast_math
NVCCFLAGS       := -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math 

ToolDAQPath=ToolDAQ

CXXFLAGS = -g -std=c++11 -fPIC -O2 -Wpedantic

ZMQLib= -L $(ToolDAQPath)/zeromq-4.0.7/lib -lzmq 
ZMQInclude= -I $(ToolDAQPath)/zeromq-4.0.7/include/ 

BoostLib= -L $(ToolDAQPath)/boost_1_66_0/install/lib -lboost_date_time -lboost_serialization -lboost_iostreams
BoostInclude= -I $(ToolDAQPath)/boost_1_66_0/install/include

DataModelInclude = $(RootInclude) $(WCSimInclude) $(BonsaiInclude)
DataModelLib =  $(RootLib) $(WCSimLib) $(BonsaiLib)

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

ifdef FLOWERDIR
	FlowerInclude = -I$(FLOWERDIR)
	FlowerLib = -L$(FLOWERDIR) -lWCSimFLOWER
endif

DOXYGEN_VERSION = $(shell doxygen --version 2>/dev/null)
ifdef DOXYGEN_VERSION
	DOXYGEN_EXISTS = 1
else
	DOXYGEN_EXISTS = 0
endif



CPU: lib/libStore.so lib/libLogging.so lib/libDataModel.so include/Tool.h lib/libMyTools.so lib/libServiceDiscovery.so lib/libToolChain.so main RemoteControl  NodeDaemon

all: lib/libStore.so lib/libLogging.so lib/libDataModel.so include/Tool.h lib/libMyTools.so lib/libServiceDiscovery.so lib/libToolChain.so main RemoteControl  NodeDaemon doxy

GPU: lib/libStore.so lib/libLogging.so lib/libDataModel.so include/Tool.h lib/libMyToolsGPU.so lib/libServiceDiscovery.so lib/libToolChain.so mainGPU RemoteControl NodeDaemon

main: src/main.cpp | lib/libMyTools.so lib/libStore.so lib/libLogging.so lib/libToolChain.so lib/libDataModel.so lib/libServiceDiscovery.so
	@echo -e "\e[38;5;226m\n*************** Making " $@ "****************\e[0m"
	g++ $(CXXFLAGS) src/main.cpp -o main -I include -L lib -lStore -lMyTools -lToolChain -lDataModel -lLogging -lServiceDiscovery -lpthread $(DataModelInclude) $(DataModelLib) $(MyToolsInclude)  $(MyToolsLib) $(ZMQLib) $(ZMQInclude)  $(BoostLib) $(BoostInclude) $(BonsaiLib) $(BonsaiInclude) $(FlowerLib) $(FlowerInclude)

mainGPU: src/main.cpp UserTools/CUDA/GPU_link.o | lib/libMyToolsGPU.so lib/libStore.so lib/libLogging.so lib/libToolChain.so lib/libDataModel.so lib/libServiceDiscovery.so
	@echo -e "\e[38;5;226m\n*************** Making " $@ "****************\e[0m"
	g++ $(CXXFLAGS) src/main.cpp UserTools/CUDA/GPU_link.o -o main -I include -L lib -lStore -lMyToolsGPU  -lMyTools -lToolChain -lDataModel -lLogging -lServiceDiscovery -lpthread $(DataModelInclude) $(DataModelLib) $(MyToolsIncludeGPU)  $(MyToolsLibGPU) $(ZMQLib) $(ZMQInclude)  $(BoostLib) $(BoostInclude)


lib/libStore.so: $(ToolDAQPath)/ToolDAQFramework/src/Store/*
	cd $(ToolDAQPath)/ToolDAQFramework && make lib/libStore.so
	@echo -e "\e[38;5;118m\n*************** Copying " $@ "****************\e[0m"
	cp $(ToolDAQPath)/ToolDAQFramework/src/Store/*.h include/
	cp $(ToolDAQPath)/ToolDAQFramework/lib/libStore.so lib/
	#g++ $(CXXFLAGS) -fPIC -shared  -I include $(ToolDAQPath)/ToolDAQFramework/src/Store/*.cpp -o lib/libStore.so $(BoostLib) $(BoostInclude)


include/Tool.h:  $(ToolDAQPath)/ToolDAQFramework/src/Tool/Tool.h
	@echo -e "\e[38;5;118m\n*************** Copying " $@ "****************\e[0m"
	cp $(ToolDAQPath)/ToolDAQFramework/src/Tool/Tool.h include/
	cp UserTools/*.h include/
	cp UserTools/*/*.h include/
	cp DataModel/*.h include/


lib/libToolChain.so: $(ToolDAQPath)/ToolDAQFramework/src/ToolChain/* | lib/libLogging.so lib/libStore.so lib/libMyTools.so lib/libServiceDiscovery.so lib/libLogging.so lib/libDataModel.so
	@echo -e "\e[38;5;226m\n*************** Making " $@ "****************\e[0m"
	cp $(ToolDAQPath)/ToolDAQFramework/UserTools/{Factory,Logger,ServiceAdd}/*.h include/
	cp $(ToolDAQPath)/ToolDAQFramework/UserTools/Unity.h include/
	cp $(ToolDAQPath)/ToolDAQFramework/src/ToolChain/*.h include/
	g++ $(CXXFLAGS) -shared $(ToolDAQPath)/ToolDAQFramework/src/ToolChain/ToolChain.cpp -I include -lpthread -L lib -lStore -lDataModel -lServiceDiscovery -lMyTools -lLogging -o lib/libToolChain.so $(DataModelInclude) $(DataModelLib) $(ZMQLib) $(ZMQInclude) $(MyToolsInclude)  $(BoostLib) $(BoostInclude)


clean: 
	@echo -e "\e[38;5;201m\n*************** Cleaning up ****************\e[0m"
	rm -f include/*.h
	rm -f lib/*.so
	rm -f main
	rm -f RemoteControl
	rm -f NodeDaemon
	rm -f UserTools/CUDA/*.o
	rm -f UserTools/*/*.o
	rm -f DataModel/*.o


lib/libDataModel.so: DataModel/* lib/libLogging.so lib/libStore.so $(patsubst DataModel/%.cpp, DataModel/%.o, $(wildcard DataModel/*.cpp))
	@echo -e "\e[38;5;226m\n*************** Making " $@ "****************\e[0m"
	cp DataModel/*.h include/
	g++ --version
	g++ $(CXXFLAGS) -shared DataModel/*.o -I include -L lib -lStore  -lLogging  -o lib/libDataModel.so $(DataModelInclude) $(DataModelLib) $(ZMQLib) $(ZMQInclude)  $(BoostLib) $(BoostInclude)

lib/libMyTools.so: UserTools/*/* UserTools/* include/Tool.h  lib/libLogging.so lib/libStore.so  $(patsubst UserTools/%.cpp, UserTools/%.o, $(wildcard UserTools/*/*.cpp)) |lib/libDataModel.so
	@echo -e "\e[38;5;226m\n*************** Making " $@ "****************\e[0m"
	cp UserTools/*/*.h include/
	cp UserTools/*.h include/
	g++ $(CXXFLAGS) -shared  UserTools/*/*.o -I include -L lib -lStore -lDataModel -lLogging -o lib/libMyTools.so $(MyToolsInclude) $(MyToolsLib) $(DataModelInclude) $(DataModelLib) $(ZMQLib) $(ZMQInclude) $(BoostLib) $(BoostInclude) $(BonsaiLib) $(BonsaiInclude) $(FlowerLib) $(FlowerInclude)

lib/libMyToolsGPU.so: UserTools/*/* UserTools/* UserTools/CUDA/GPU_link.o include/Tool.h lib/libLogging.so lib/libStore.so  $(patsubst UserTools/%.cpp, UserTools/%.o, $(wildcard UserTools/*/*.cpp)) | lib/libDataModel.so
	@echo -e "\e[38;5;226m\n*************** Making " $@ "****************\e[0m"
	cp UserTools/*/*.h include/
	cp UserTools/*.h include/
	g++ $(CXXFLAGS) -shared  UserTools/*/*.o  -DGPU UserTools/CUDA/CUDA_Unity.o -I include -L lib -lStore -lDataModel -lLogging -o lib/libMyToolsGPU.so $(MyToolsIncludeGPU) $(MyToolsLibGPU) $(DataModelInclude) $(DataModelLib) $(ZMQLib) $(ZMQInclude) $(BoostLib) $(BoostInclude)

RemoteControl:
	cd $(ToolDAQPath)/ToolDAQFramework/ && make RemoteControl
	@echo -e "\e[38;5;118m\n*************** Copying " $@ "****************\e[0m"
	cp $(ToolDAQPath)/ToolDAQFramework/RemoteControl ./

NodeDaemon: 
	cd $(ToolDAQPath)/ToolDAQFramework/ && make NodeDaemon
	@echo -e "\e[38;5;226m\n*************** Copying " $@ "****************\e[0m"
	cp $(ToolDAQPath)/ToolDAQFramework/NodeDaemon ./

lib/libServiceDiscovery.so: $(ToolDAQPath)/ToolDAQFramework/src/ServiceDiscovery/* | lib/libStore.so
	cd $(ToolDAQPath)/ToolDAQFramework && make lib/libServiceDiscovery.so
	@echo -e "\e[38;5;118m\n*************** Copying " $@ "****************\e[0m"
	cp $(ToolDAQPath)/ToolDAQFramework/src/ServiceDiscovery/ServiceDiscovery.h include/
	cp $(ToolDAQPath)/ToolDAQFramework/lib/libServiceDiscovery.so lib/
	#g++ $(CXXFLAGS) -shared -fPIC -I include $(ToolDAQPath)/ToolDAQFramework/src/ServiceDiscovery/ServiceDiscovery.cpp -o lib/libServiceDiscovery.so -L lib/ -lStore  $(ZMQInclude) $(ZMQLib) $(BoostLib) $(BoostInclude)

lib/libLogging.so:  $(ToolDAQPath)/ToolDAQFramework/src/Logging/* | lib/libStore.so
	cd $(ToolDAQPath)/ToolDAQFramework && make lib/libLogging.so
	@echo -e "\e[38;5;118m\n*************** Copying " $@ "****************\e[0m"
	cp $(ToolDAQPath)/ToolDAQFramework/src/Logging/Logging.h include/
	cp $(ToolDAQPath)/ToolDAQFramework/lib/libLogging.so lib/
	#g++ $(CXXFLAGS) -shared -fPIC -I include $(ToolDAQPath)/ToolDAQFramework/src/Logging/Logging.cpp -o lib/libLogging.so -L lib/ -lStore $(ZMQInclude) $(ZMQLib) $(BoostLib) $(BoostInclude)

update:
	@echo -e "\e[38;5;51m\n*************** Updating ****************\e[0m"
	cd $(ToolDAQPath)/ToolDAQFramework; git pull
	cd $(ToolDAQPath)/zeromq-4.0.7; git pull
	git pull

UserTools/CUDA/GPU_link.o:  UserTools/CUDA/*
	@echo "\n*************** Compiling & Linking " $@ "****************"
	cp UserTools/CUDA/*.h include/
	nvcc -c --shared -Xcompiler -fPIC -dlink UserTools/CUDA/CUDA_Unity.cu -o UserTools/CUDA/CUDA_Unity.o  -I UserTools/CUDA/  $(CUDAINC) $(NVCCFLAGS) $(CUDALIB)
	nvcc  -arch=sm_35 -dlink  -o UserTools/CUDA/GPU_link.o UserTools/CUDA/CUDA_Unity.o  $(CUDALIB) $(CUDAINC)

Docs:
	@if [ ${DOXYGEN_EXISTS} = 1 ]; \
	then \
		doxygen docs/doxygen.config; \
	else \
		echo "Error: doxygen program not found in path. Exiting"; \
	fi

UserTools/%.o: UserTools/%.cpp lib/libStore.so include/Tool.h lib/libLogging.so lib/libDataModel.so
	@echo -e "\e[38;5;226m\n*************** Making " $@ "****************\e[0m"
	cp $(shell dirname $<)/*.h include
	-g++ -c $(CXXFLAGS) -o $@ $< -I include -L lib -lStore -lDataModel -lLogging $(MyToolsInclude) $(MyToolsLib) $(DataModelInclude) $(DataModelib) $(ZMQLib) $(ZMQInclude) $(BoostLib) $(BoostInclude) $(BonsaiLib) $(BonsaiInclude) $(FlowerLib) $(FlowerInclude)

target: remove $(patsubst %.cpp, %.o, $(wildcard UserTools/$(TOOL)/*.cpp))

remove:
	echo -e "removing"
	-rm UserTools/$(TOOL)/*.o

DataModel/%.o: DataModel/%.cpp lib/libLogging.so lib/libStore.so
	@echo -e "\e[38;5;226m\n*************** Making " $@ "****************\e[0m"
	cp $(shell dirname $<)/*.h include
	-g++ -c $(CXXFLAGS) -o $@ $< -I include -L lib -lStore -lLogging  $(DataModelInclude) $(DataModelLib) $(ZMQLib) $(ZMQInclude) $(BoostLib) $(BoostInclude)
