#Compiler
#g++ --std=c++11 -no-pie -Iincludes parser.cpp lib/main-parser.o lib/lib.a
CC=g++ -no-pie -g

#Compiler flags
CFLAGS= -c `pkg-config opencv --cflags` -g3 #-fPIC
#Linker flags
#LDFLAGS = `pkg-config opencv --cflags --libs`
LDFLAGS= `pkg-config opencv --libs` #-fPIC
#export CXXFLAGS="$CXXFLAGS -fPIC"

#Libraries to be included 
#LIBS =
LIBS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lcurl
#Include path
#INC = 
INC = -I/usr/local/include/opencv -I/usr/local/include/

#Source files
#SOURCES=iSTAB.cpp Frame.cpp Capture.cpp HistoryList.cpp
SOURCES=main.cpp motion.cpp Kalman.cpp medianFilter.cpp
#HEADERS=iSTAB.h Frame.h Capture.h HistoryList.h
HEADERS=motion.cpp Kalman.h medianFilter.h Mediator.h

#Creates object files, No need to do anything
OBJECTS=$(SOURCES:.cpp=.o)

#Write name of final executable file
EXECUTABLE=main

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) $(HEADERS)
	$(CC) $(OBJECTS) $(LIBS)  -o $@ $(LDFLAGS)

.cpp.o: 
	$(CC) $(CFLAGS)  $< -o $@ $(LIBS) $(INC) 
clean:
	rm -f $(OBJECTS) $(EXECUTABLE) 
