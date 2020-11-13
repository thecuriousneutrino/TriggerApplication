#include <TimeSlice.h>

TimeSlice::TimeSlice() : chunks(0) {}

TimeSlice::~TimeSlice(){

delete chunks;
chunks=0;

}
